# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimization for BART model"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from transformers.configuration_auto import BartConfig
from transformers.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
from transformers.modeling_bart import (BartForConditionalGeneration,
                                        SelfAttention, _reorder_buffer)

from fastseq.logging import get_logger
from fastseq.utils.api_decorator import replace

logger = get_logger(__name__)

@replace(SelfAttention)
class SelfAttentionV2(SelfAttention):
    """"
    The BART Model with a language modeling head. Can be used for summarization.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
        num_beams=1,
    ):
        super().__init__(
            embed_dim, num_heads, dropout, bias, encoder_decoder_attention)
        self.num_beams = num_beams

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions=False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state and static_kv:
                # previous time steps are cached - no need to recompute key and
                # value if they are static
                key = None
        else:
            saved_state = None
            layer_state = {}
        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(
                k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        cache_bsz = (bsz // self.num_beams
                     if self.encoder_decoder_attention else bsz)

        if self.encoder_decoder_attention and ("prev_key" not in saved_state):
            cache_shape = (
                cache_bsz, self.num_beams, self.num_heads, -1, self.head_dim)
            k = k.view(cache_shape)[:, 0 : 1, :, :, :].contiguous()
            v = v.view(cache_shape)[:, 0 : 1, :, :, :].contiguous()
            layer_state[self.cache_key] = {
                "prev_key": k,
                "prev_value": v,
                "prev_key_padding_mask":
                key_padding_mask if not static_kv else None,
            }
        if not self.encoder_decoder_attention:
            cache_shape = (bsz, self.num_heads, -1, self.head_dim)
            layer_state[self.cache_key] = {
                "prev_key": k.view(cache_shape),
                "prev_value": v.view(cache_shape),
                "prev_key_padding_mask":
                key_padding_mask if not static_kv else None,
            }

        assert k is not None
        if self.encoder_decoder_attention:
            q = q.view(cache_bsz, self.num_beams, self.num_heads, tgt_len,
                       self.head_dim)
            src_len = k.size(3)
            attn_weights = torch.einsum("bmhtd,bnhsd->bmhts", q,
                                        k).reshape(-1, tgt_len, src_len)
            assert attn_weights.size() == (bsz * self.num_heads, tgt_len,
                                           src_len)
        else:
            src_len = k.size(1)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            assert attn_weights.size() == (bsz * self.num_heads, tgt_len,
                                           src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                                             src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                                             src_len)

        # This is part of a workaround to get around fork/join parallelism not
        # supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (
            bsz, src_len)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                                             src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                                             src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )

        assert v is not None
        if self.encoder_decoder_attention:
            attn_probs = attn_probs.view(
                cache_bsz, self.num_beams, self.num_heads, tgt_len, src_len)
            attn_output = torch.einsum("bmhts,bnhsd->bmhtd", attn_probs,
                                       v).reshape(-1, tgt_len, self.head_dim)
        else:
            attn_output = torch.bmm(attn_probs, v)

        assert attn_output.size() == (bsz * self.num_heads, tgt_len,
                                      self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(
            tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        if output_attentions:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                                             src_len)
        else:
            attn_weights = None
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv,
                         bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        # note that for self-attn, bsz=input_bsz * beam_size; for
        # encoder-decoder-attn, bsz=input_bsz.
        if "prev_key" in saved_state:
            prev_key_ = saved_state["prev_key"]
            if static_kv:
                k = prev_key_
            else:
                assert prev_key_ is not None
                prev_key = prev_key_.view(bsz * self.num_heads, -1,
                                          self.head_dim)
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)

        if "prev_value" in saved_state:
            prev_value_ = saved_state["prev_value"]
            assert prev_value_ is not None
            if static_kv:
                v = prev_value_
            else:
                prev_value = prev_value_.view(bsz * self.num_heads, -1,
                                              self.head_dim)
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)

        assert k is not None and v is not None
        prev_key_padding_mask: Optional[Tensor] = saved_state.get(
            "prev_key_padding_mask", None)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = key_padding_mask
        return k, v, new_key_padding_mask


@replace(BartForConditionalGeneration)
class BartForConditionalGenerationV2(BartForConditionalGeneration):
    """
    The BART Model with a language modeling head. Can be used for
    summarization.
    """
    @staticmethod
    def _reorder_cache(past, beam_idx):
        (encoder_outputs, decoder_past_key_values) = past
        enc_out, enc_mask = encoder_outputs[:2]
        reordered_past = decoder_past_key_values
        if decoder_past_key_values is not None:
            reordered_past = []
            for layer_past in decoder_past_key_values:
                # Get the correct batch idx from decoder layer's batch dim for
                # self-attn; Note that there is no need to reorder the cached
                # key and value for the encoder-decoder-attn, because the key
                # and value for the beams of each sample is the same and we can
                # cache just one copy to save GPU memory.
                layer_past_new = {}
                for attn_key, attn_cache in layer_past.items():
                    if attn_key == 'self':
                        layer_past_new[attn_key] = _reorder_buffer(
                            attn_cache, beam_idx)
                        continue
                    layer_past_new[attn_key] = attn_cache

                reordered_past.append(layer_past_new)

        new_enc_mask = (enc_mask if (enc_mask is None or enc_mask == [])
            else enc_mask.index_select(0, beam_idx))

        past = ((enc_out, new_enc_mask, *encoder_outputs[2:]), reordered_past)
        return past

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[BartConfig] = BartForConditionalGenerationV2 # pylint: disable=line-too-long
