# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Apply the beam search optimizations to fairseq-latest"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq import utils
from fairseq.models.fairseq_encoder import EncoderOut
from fairseq.models.transformer import TransformerEncoder, TransformerModel
from fairseq.modules.multihead_attention import MultiheadAttention
from fairseq.sequence_generator import SequenceGenerator

from fastseq.utils.api_decorator import register_fairseq_optimized_class, replace

@register_fairseq_optimized_class
@replace(TransformerEncoder)
class TransformerEncoderV2(TransformerEncoder):
    """
    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """
    def _reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        return encoder_out


@register_fairseq_optimized_class
@replace(TransformerModel)
class TransformerModelV2(TransformerModel):
    """ Represent the BART model."""

    def make_generation_fast_(self, **kwargs):
        super().make_generation_fast_(**kwargs)  # pylint: disable=bad-super-call
        # Replace reorder_encoder_out with a dummy function.
        if ('beamable_mm_beam_size' in kwargs and
            kwargs['beamable_mm_beam_size'] > 1):
            self.encoder.reorder_encoder_out = self.encoder._reorder_encoder_out


@register_fairseq_optimized_class
@replace(MultiheadAttention)
class MultiheadAttentionV2(MultiheadAttention):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        kdim=None,
        vdim=None,
        dropout=0.0,
        bias=True,
        add_bias_kv=False,
        add_zero_attn=False,
        self_attention=False,
        encoder_decoder_attention=False,
        q_noise=0.0,
        qn_block_size=8,
    ):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout, bias,
                         add_bias_kv, add_zero_attn, self_attention,
                         encoder_decoder_attention, q_noise, qn_block_size)
        self.dropout = dropout
        self.beam_size = 1

    def forward(
        self,
        query,
        key: Optional[Tensor],
        value: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str,
                                                   Optional[Tensor]]]] = None,
        need_weights: bool = True,
        static_kv: bool = False,
        attn_mask: Optional[Tensor] = None,
        before_softmax: bool = False,
        need_head_weights: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time x Batch x Channel
        Args:
            key_padding_mask (ByteTensor, optional): mask to exclude
                keys that are pads, of shape `(batch, src_len)`, where
                padding elements are indicated by 1s.
            need_weights (bool, optional): return the attention weights,
                averaged over heads (default: False).
            attn_mask (ByteTensor, optional): typically used to
                implement causal attention, where the mask prevents the
                attention from looking forward in time (default: None).
            before_softmax (bool, optional): return the raw attention
                weights and values before the attention softmax.
            need_head_weights (bool, optional): return the attention
                weights for each head. Implies *need_weights*. Default:
                return the average attention weights over all heads.
        """
        if need_head_weights:
            need_weights = True
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        if (not self.onnx_trace
            and not self.tpu  # don't use PyTorch version on TPUs
            and incremental_state is None and not static_kv
            # A workaround for quantization to work. Otherwise JIT compilation
            # treats bias in linear module as method.
            and not torch.jit.is_scripting()):
            assert key is not None and value is not None
            return F.multi_head_attention_forward(
                query,
                key,
                value,
                self.embed_dim,
                self.num_heads,
                torch.empty([0]),
                torch.cat(
                    (self.q_proj.bias, self.k_proj.bias, self.v_proj.bias)),
                self.bias_k,
                self.bias_v,
                self.add_zero_attn,
                self.dropout,
                self.out_proj.weight,
                self.out_proj.bias,
                self.training,
                key_padding_mask,
                need_weights,
                attn_mask,
                use_separate_proj_weight=True,
                q_proj_weight=self.q_proj.weight,
                k_proj_weight=self.k_proj.weight,
                v_proj_weight=self.v_proj.weight,
            )
        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if saved_state is not None and "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute
                # key and value if they are static
                if static_kv:
                    assert (self.encoder_decoder_attention and
                            not self.self_attention)
                    key = value = None
        else:
            saved_state = None
        if self.self_attention:
            q = self.q_proj(query)
            k = self.k_proj(query)
            v = self.v_proj(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.q_proj(query)
            if key is None:
                assert value is None
                k = v = None
            else:
                if self.beam_size > 1 and bsz == key.size(1):
                    # key is [T, bsz*beam_size, C], reduce to [T, bsz, C]
                    key = key.view(key.size(0), -1, self.beam_size,
                                   key.size(2))[:, :, 0, :]
                    if key_padding_mask is not None:
                        key_padding_mask = key_padding_mask.view(
                            -1, self.beam_size, key_padding_mask.size(1))[:,
                                                                          0, :]
                k = self.k_proj(key)
                v = self.v_proj(key)

        else:
            assert key is not None and value is not None
            q = self.q_proj(query)
            k = self.k_proj(key)
            v = self.v_proj(value)
        q *= self.scaling
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask,
                     attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        key_padding_mask.new_zeros(key_padding_mask.size(0),
                                                   1),
                    ],
                    dim=1,
                )
        q = (q.contiguous().view(tgt_len, bsz * self.num_heads,
                                 self.head_dim).transpose(0, 1))
        if k is not None:
            kv_bsz = k.size(1)
            k = (k.contiguous().view(-1, kv_bsz * self.num_heads,
                                     self.head_dim).transpose(0, 1))
        if v is not None:
            assert kv_bsz
            v = (v.contiguous().view(-1, kv_bsz * self.num_heads,
                                     self.head_dim).transpose(0, 1))

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if "prev_key" in saved_state:
                saved_prev_key = saved_state["prev_key"]
                assert saved_prev_key is not None
                kv_bsz = saved_prev_key.size(0)
                prev_key = saved_prev_key.view(kv_bsz * self.num_heads, -1,
                                               self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat([prev_key, k], dim=1)
            if "prev_value" in saved_state:
                saved_prev_value = saved_state["prev_value"]
                assert saved_prev_value is not None
                assert kv_bsz == saved_prev_value.size(0)
                prev_value = saved_prev_value.view(kv_bsz * self.num_heads, -1,
                                                   self.head_dim)
                if static_kv:
                    v = prev_value
                else:
                    assert v is not None
                    v = torch.cat([prev_value, v], dim=1)
            prev_key_padding_mask: Optional[Tensor] = None
            if "prev_key_padding_mask" in saved_state:
                prev_key_padding_mask = saved_state["prev_key_padding_mask"]
            assert k is not None and v is not None
            key_padding_mask = MultiheadAttention._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=kv_bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )
            saved_state["prev_key"] = k.view(kv_bsz, self.num_heads, -1,
                                             self.head_dim)
            saved_state["prev_value"] = v.view(kv_bsz, self.num_heads, -1,
                                               self.head_dim)
            saved_state["prev_key_padding_mask"] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            incremental_state = self._set_input_buffer(incremental_state,
                                                       saved_state)
        assert k is not None
        src_len = k.size(1)
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None

        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == kv_bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            assert v is not None
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])],
                          dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])],
                          dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat(
                    [attn_mask,
                     attn_mask.new_zeros(attn_mask.size(0), 1)],
                    dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [
                        key_padding_mask,
                        torch.zeros(key_padding_mask.size(0),
                                    1).type_as(key_padding_mask),
                    ],
                    dim=1,
                )

        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn_weights = torch.einsum(
                'bxhtd,bhsd->bxhts',
                q.view(kv_bsz, -1, self.num_heads,
                       *q.size()[1:]),
                k.view(kv_bsz, self.num_heads,
                       *k.size()[1:]))
            attn_weights = attn_weights.reshape(-1, *attn_weights.size()[-2:])
        else:
            attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = MultiheadAttention.apply_sparse_mask(
            attn_weights, tgt_len, src_len, bsz)

        assert list(
            attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask
        if key_padding_mask is not None:
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len,
                                             src_len)
            if not self.tpu:
                attn_weights = attn_weights.view(kv_bsz, -1, self.num_heads,
                                                 tgt_len, src_len)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2).unsqueeze(3).to(
                        torch.bool), float("-inf"))
            else:
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                                             src_len)

        if before_softmax:
            return attn_weights, v
        attn_weights_float = utils.softmax(attn_weights,
                                           dim=-1,
                                           onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(
            attn_weights,
            p=self.dropout,
            training=self.training,
        )
        assert v is not None
        if self.encoder_decoder_attention and bsz != kv_bsz:
            attn = torch.einsum(
                'bxhts,bhsd->bxhtd',
                attn_probs.view(kv_bsz, -1, self.num_heads,
                                *attn_probs.size()[1:]),
                v.view(kv_bsz, self.num_heads,
                       *v.size()[1:]))
            attn = attn.reshape(-1, *attn.size()[-2:])
        else:
            attn = torch.bmm(attn_probs, v)
        assert list(
            attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if self.onnx_trace and attn.size(1) == 1:
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0,
                                  1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)
        attn_weights: Optional[Tensor] = None
        if need_weights:
            attn_weights = attn_weights_float.view(bsz, self.num_heads,
                                                   tgt_len,
                                                   src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        return attn, attn_weights

    @torch.jit.export
    def reorder_incremental_state(
        self, incremental_state: Dict[str, Dict[str, Optional[Tensor]]],
        new_order: Tensor):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer_k is not None:
                    if self.encoder_decoder_attention:
                        if input_buffer_k.size(
                            0) * self.beam_size == new_order.size(0):
                            return incremental_state
                        elif self.beam_size > 1:
                            input_buffer[k] = input_buffer_k.index_select(
                                0,
                                new_order.reshape(-1, self.beam_size)[:, 0] //
                                self.beam_size)
                        else:
                            input_buffer[k] = input_buffer_k.index_select(
                                0, new_order)
                    else:
                        input_buffer[k] = input_buffer_k.index_select(
                            0, new_order)
            incremental_state = self._set_input_buffer(incremental_state,
                                                       input_buffer)
        return incremental_state

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size

    def make_generation_fast_(self, beamable_mm_beam_size=None, **kwargs):
        if beamable_mm_beam_size is not None:
            self.set_beam_size(beamable_mm_beam_size)


@register_fairseq_optimized_class
@replace(SequenceGenerator)
class SequenceGeneratorV2(SequenceGenerator):
    """
    Sequence Generator is optimized by reducing the cached memory usage
    during the encoding period for beam search.
    """

    def _no_repeat_ngram(self, tokens, lprobs, bsz: int, beam_size: int,
                         step: int):
        # for each beam and batch sentence, generate a list of previous ngrams
        gen_ngrams: List[Dict[str, List[int]]] = [
            torch.jit.annotate(Dict[str, List[int]], {})
            for bbsz_idx in range(bsz * beam_size)
        ]
        cpu_tokens = tokens.cpu()[:, :step + 1]
        for bbsz_idx in range(bsz * beam_size):
            gen_tokens = cpu_tokens[bbsz_idx].tolist()
            for ngram in zip(*[
                gen_tokens[i:]
                for i in range(self.no_repeat_ngram_size)
            ]):
                if ngram[-1] != self.pad:
                    gen_ngrams[bbsz_idx][tuple(ngram[:-1])] = \
                        gen_ngrams[bbsz_idx].get(tuple(ngram[:-1]), [])\
                        + [ngram[-1]]

        banned_tokens = []
        if step + 2 - self.no_repeat_ngram_size >= 0:
            # no banned tokens if we haven't generated no_repeat_ngram_size tokens yet
            banned_tokens = [
                self.calculate_banned_tokens(cpu_tokens, step, gen_ngrams,
                                             self.no_repeat_ngram_size,
                                             bbsz_idx)
                for bbsz_idx in range(bsz * beam_size)
            ]
        else:
            banned_tokens = [
                torch.jit.annotate(List[int], [])
                for bbsz_idx in range(bsz * beam_size)
            ]
        banned_lprobs = [(bbsz_idx, banned_idx)
                         for bbsz_idx in range(len(banned_tokens))
                         for banned_idx in banned_tokens[bbsz_idx]]
        if banned_lprobs:
            banned_lprobs = tuple(torch.LongTensor(list(zip(*banned_lprobs))))
            lprobs.index_put_(
                banned_lprobs,
                lprobs.new_tensor([-math.inf] * banned_lprobs[0].nelement()))
        return lprobs
