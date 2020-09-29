# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""Optimization for T5 model"""

import logging

import torch
import torch.nn.functional as F

from fastseq.utils.api_decorator import replace
from transformers.configuration_t5 import T5Config
from transformers.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
from transformers.modeling_t5 import T5Attention, T5ForConditionalGeneration

logger = logging.getLogger(__name__)


@replace(T5Attention)
class T5AttentionV2(T5Attention):
    """Optimized T5Attention for self-attn and encoder-decoder-attn in T5."""

    def __init__(self,
                 config: T5Config,
                 has_relative_attention_bias=False,
                 num_beams=1):
        super().__init__(
            config=config,
            has_relative_attention_bias=has_relative_attention_bias)
        self.num_beams = num_beams

    def forward(
        self,
        input,
        mask=None,
        kv=None,
        position_bias=None,
        past_key_value_state=None,
        head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
    ):
        """
        Self-attention (if kv is None) or attention over source sentence
        (provided by kv).
        """
        # Input is (bs, qlen, dim)
        # Mask is (bs, klen) (non-causal) or (bs, klen, klen)
        # past_key_value_state[0] is (bs, n_heads, q_len - 1, dim_per_head)
        bs, qlen, dim = input.size()

        is_encoder_decoder_attn = kv is not None

        if past_key_value_state is not None:
            assert (self.is_decoder is
                    True), "Encoder cannot cache past key value states"
            assert (
                len(past_key_value_state) == 2
            ), "past_key_value_state should have 2 past states: keys and values"
            ". Got {} past states".format(len(past_key_value_state))
            real_qlen = qlen + past_key_value_state[0].shape[
                2] if query_length is None else query_length
        else:
            real_qlen = qlen

        if kv is None:
            klen = real_qlen
        else:
            klen = kv.size(1)

        def shape(x):
            """  projection """
            return x.view(bs, -1, self.n_heads, self.d_kv).transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous().view(bs, -1, self.inner_dim)

        q = shape(self.q(input))  # (bs, n_heads, qlen, dim_per_head)

        if kv is None:
            k = shape(self.k(input))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(input))  # (bs, n_heads, qlen, dim_per_head)
        elif past_key_value_state is None:
            k = v = kv
            k = shape(self.k(k))  # (bs, n_heads, qlen, dim_per_head)
            v = shape(self.v(v))  # (bs, n_heads, qlen, dim_per_head)

        if past_key_value_state is not None:
            if kv is None:
                k_, v_ = past_key_value_state
                k = torch.cat(
                    [k_, k], dim=2)  # (bs, n_heads, klen, dim_per_head)
                v = torch.cat(
                    [v_, v], dim=2)  # (bs, n_heads, klen, dim_per_head)
            else:
                k, v = past_key_value_state

        if self.is_decoder and use_cache is True:
            if is_encoder_decoder_attn:
                if past_key_value_state is None:
                    k = k.view(bs // self.num_beams, self.num_beams,
                               self.n_heads, klen,
                               self.d_kv)[:, 0:1, :, :, :].contiguous()
                    v = v.view(bs // self.num_beams, self.num_beams,
                               self.n_heads, klen,
                               self.d_kv)[:, 0:1, :, :, :].contiguous()
            present_key_value_state = ((k, v),)
        else:
            present_key_value_state = (None,)

        if is_encoder_decoder_attn and use_cache:
            new_q = q.view(bs // self.num_beams, self.num_beams, self.n_heads,
                           qlen, self.d_kv)
            scores = torch.einsum(
                "bmnqd,bxnkd->bmnqk", new_q, k).reshape(
                    -1, self.n_heads, qlen, klen)  # (bs, n_heads, qlen, klen)
        else:
            scores = torch.einsum(
                "bnqd,bnkd->bnqk", q, k)  # (bs, n_heads, qlen, klen)

        if position_bias is None:
            if not self.has_relative_attention_bias:
                raise ValueError(
                    "No position_bias provided and no weights to compute"
                    "position_bias")
            position_bias = self.compute_bias(real_qlen, klen)

            # if key and values are already calculated
            # we want only the last query position bias
            if past_key_value_state is not None:
                position_bias = position_bias[:, :, -1:, :]

            if mask is not None:
                position_bias = position_bias + mask # (bs, n_heads, qlen, klen)

        scores += position_bias
        weights = F.softmax(scores.float(), dim=-1).type_as(
            scores)  # (bs, n_heads, qlen, klen)
        weights = F.dropout(
            weights, p=self.dropout,
            training=self.training)  # (bs, n_heads, qlen, klen)

        # Mask heads if we want to
        if head_mask is not None:
            weights = weights * head_mask
        if is_encoder_decoder_attn and use_cache:
            tmp_weights = weights.view(bs // self.num_beams, self.num_beams,
                                       self.n_heads, qlen, klen)
            context = torch.einsum(
                "bmnqk,bxnkd->bmnqd", tmp_weights, v).reshape(
                    -1, self.n_heads, qlen, self.d_kv
                    )  # (bs, n_heads, qlen, dim_per_head)
        else:
            context = torch.matmul(
                weights, v)  # (bs, n_heads, qlen, dim_per_head)
        context = unshape(context)  # (bs, qlen, dim)

        context = self.o(context)

        outputs = (context,) + present_key_value_state

        if output_attentions:
            outputs = outputs + (weights,)
        if self.has_relative_attention_bias:
            outputs = outputs + (position_bias,)
        return outputs


@replace(T5ForConditionalGeneration)
class T5ForConditionalGenerationV2(T5ForConditionalGeneration):
    """Optimized T5ForConditionalGenerationV2"""

    def _reorder_cache(self, past, beam_idx):
        # if decoder past is not included in output
        # speedy decoding is disabled and no need to reorder
        if past[1] is None:
            logger.warning(
                "You might want to consider setting `use_cache=True` to speed"
                "up decoding")
            return past

        decoder_past = past[1]
        past = (past[0],)
        reordered_decoder_past = ()
        for layer_past_states in decoder_past:
            # get the correct batch idx from layer past batch dim
            # batch dim of `past` is at 2nd position
            reordered_layer_past_states = ()
            for layer_past_state in layer_past_states[0:2]:
                # need to set correct `past` for each of the four key / value
                # states
                reordered_layer_past_states = reordered_layer_past_states + (
                    layer_past_state.index_select(0, beam_idx),)
            reordered_layer_past_states = (
                reordered_layer_past_states + layer_past_states[2:])

            assert reordered_layer_past_states[0].shape == layer_past_states[
                0].shape
            assert len(reordered_layer_past_states) == len(layer_past_states)

            reordered_decoder_past = reordered_decoder_past + (
                reordered_layer_past_states,)
        return past + (reordered_decoder_past,)

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[T5Config] = T5ForConditionalGenerationV2  # pylint: disable=line-too-long
