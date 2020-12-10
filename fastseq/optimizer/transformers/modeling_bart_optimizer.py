# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimization for BART model"""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from transformers.activations import ACT2FN
from transformers.configuration_auto import BartConfig
from transformers.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
from transformers.modeling_bart import (BartForConditionalGeneration,
                                        DecoderLayer, EncoderLayer, LayerNorm,
                                        SelfAttention, _reorder_buffer)

from fastseq.logging import get_logger
from fastseq.utils.api_decorator import replace
from fastseq.optimizer.jit.graph_rewriter import optimize_graph

logger = get_logger(__name__)

@replace(SelfAttention)
class SelfAttentionV2(nn.Module):
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
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, (
            "embed_dim must be divisible by num_heads")
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention: bool = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if (
            self.encoder_decoder_attention) else "self"
        self.num_beams = num_beams

    def _shape(self, tensor: Tensor, dim_0: int, bsz: int) -> Tensor:
        return tensor.contiguous().view(
            dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query: Tensor,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        attn_mask: Optional[Tensor] = None,
        output_attentions: bool=False,
    ) -> Tuple[Tensor,
               Optional[Tensor],
               Optional[Dict[str, Dict[str, Optional[Tensor]]]]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv: bool = self.encoder_decoder_attention
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        saved_state: Dict[str, Optional[Tensor]] = {}
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            if self.cache_key in layer_state:
                tmp_saved_state = layer_state.get(self.cache_key)
                assert tmp_saved_state is not None
                saved_state = tmp_saved_state
            if static_kv and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute key and
                # value if they are static
                key = None

        q = self.q_proj(query) * self.scaling
        q = self._shape(q, tgt_len, bsz)

        k: Optional[Tensor] = None
        v: Optional[Tensor] = None
        if key is not None:
            k = self.k_proj(key)
            k = self._shape(k, -1, bsz)
            v = self.v_proj(key)
            v = self._shape(v, -1, bsz)

        if len(saved_state) > 0:
            k, v, key_padding_mask = self._use_saved_state(
                k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        cache_bsz = (bsz // self.num_beams
                     if self.encoder_decoder_attention else bsz)

        assert k is not None
        assert v is not None
        if self.encoder_decoder_attention and ("prev_key" not in saved_state):
            cache_shape = (
                cache_bsz, self.num_beams, self.num_heads, -1, self.head_dim)
            k = k.view(cache_shape)[:, 0 : 1, :, :, :].contiguous()
            v = v.view(cache_shape)[:, 0 : 1, :, :, :].contiguous()
            prev_k: Optional[Tensor] = k
            prev_v: Optional[Tensor] = v
            prev_key_padding_mask: Optional[Tensor] = None if (
                static_kv) else key_padding_mask
            assert layer_state is not None
            layer_state[self.cache_key] = {
                "prev_key": prev_k,
                "prev_value": prev_v,
                "prev_key_padding_mask": prev_key_padding_mask,
            }

        if not self.encoder_decoder_attention and layer_state is not None:
            cache_shape = (bsz, self.num_heads, -1, self.head_dim)
            prev_k: Optional[Tensor] = k.view(cache_shape)
            prev_v: Optional[Tensor] = v.view(cache_shape)
            prev_key_padding_mask: Optional[Tensor] = None if (
                static_kv) else key_padding_mask
            assert layer_state is not None
            layer_state[self.cache_key] = {
                "prev_key": prev_k,
                "prev_value": prev_v,
                "prev_key_padding_mask": prev_key_padding_mask,
            }

        # assert q is not None
        if self.encoder_decoder_attention:
            q = q.view(cache_bsz, self.num_beams, self.num_heads, tgt_len,
                       self.head_dim)
            src_len = k.size(3)
            attn_weights = torch.einsum("bmhtd,bnhsd->bmhts", q, k).reshape(
                -1, tgt_len, src_len)
            assert attn_weights.size() == (
                bsz * self.num_heads, tgt_len, src_len)
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
            return attn_output, attn_weights, layer_state
        else:
            return attn_output, None, layer_state

    def _use_saved_state(
        self,
        k: Optional[Tensor],
        v: Optional[Tensor],
        saved_state: Dict[str, Optional[Tensor]],
        key_padding_mask: Optional[Tensor],
        static_kv: bool,
        bsz: int) -> Tuple[
            Optional[Tensor], Optional[Tensor], Optional[Tensor]]:
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
            "prev_key_padding_mask")
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                assert prev_key_padding_mask is not None
                assert key_padding_mask is not None
                new_key_padding_mask = torch.cat(
                    [prev_key_padding_mask, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = key_padding_mask
        return k, v, new_key_padding_mask


@replace(EncoderLayer)
class EncoderLayerV2(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = SelfAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        self.self_attn = torch.jit.script(self.self_attn)
        optimize_graph(self.self_attn.graph)
        self.normalize_before = config.normalize_before
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask, output_attentions=False):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, attn_weights, layer_state = self.self_attn(
            query=x,
            key=x,
            key_padding_mask=encoder_padding_mask,
            output_attentions=output_attentions,
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, attn_weights


@replace(DecoderLayer)
class DecoderLayerV2(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout,
        )
        self.self_attn = torch.jit.script(self.self_attn)
        optimize_graph(self.self_attn.graph)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.normalize_before = config.normalize_before

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn = torch.jit.script(self.encoder_attn)
        optimize_graph(self.encoder_attn.graph)
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self,
        x,
        encoder_hidden_states,
        encoder_attn_mask=None,
        layer_state=None,
        causal_mask=None,
        decoder_padding_mask=None,
        output_attentions=False,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        # Self Attention

        x, self_attn_weights, layer_state = self.self_attn(
            query=x,
            key=x,
            layer_state=layer_state,  # adds keys to layer state
            key_padding_mask=decoder_padding_mask,
            attn_mask=causal_mask,
            output_attentions=output_attentions,
        )

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        # Cross attention
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key
        if self.normalize_before:
            x = self.encoder_attn_layer_norm(x)
        x, attn_weights, layer_state = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.encoder_attn_layer_norm(x)

        # Fully Connected
        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


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
