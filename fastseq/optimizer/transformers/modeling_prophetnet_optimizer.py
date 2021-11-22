# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimization for ProphetNet model"""

from typing import Optional, Tuple
import torch
from torch import Tensor, nn
from transformers.models.prophetnet.configuration_prophetnet import ProphetNetConfig

from fastseq.utils.api_decorator import replace
from transformers.models.prophetnet.modeling_prophetnet import ProphetNetAttention


@replace(ProphetNetAttention)
class ProphetNetAttentionV2(ProphetNetAttention):
    def __init__(
        self, 
        config: ProphetNetConfig, 
        num_attn_heads: int,
        num_beams: int = 1
    ):
        super().__init__(config, num_attn_heads)
        self.num_beams = num_beams
        
    def forward(
        self,
        hidden_states,
        key_value_states: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        layer_head_mask: Optional[Tensor] = None,
        past_key_value: Optional[Tuple[Tensor]] = None,
        output_attentions: bool = False,
    ) -> Tuple[Tensor, Optional[Tensor]]:

        batch_size, tgt_len, hidden_size = hidden_states.size()

        # if key_value_states are provided this layer is used as a cross-attention layer
        # for the decoder
        is_cross_attention = key_value_states is not None
        assert list(hidden_states.size()) == [
            batch_size,
            tgt_len,
            hidden_size,
        ], f"Size of hidden states should be {batch_size, tgt_len, hidden_size}, but is {hidden_states.size()}"

        # previous time steps are cached - no need to recompute key and value if they are static
        query_states = self.query_proj(hidden_states) / (self.head_dim ** 0.5)

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_states = past_key_value[0]
            value_states = past_key_value[1]
        elif is_cross_attention:
            # cross_attentions
            key_states = self._shape(self.key_proj(key_value_states), -1, batch_size)
            value_states = self._shape(self.value_proj(key_value_states), -1, batch_size)
        else:
            # self_attention
            key_states = self._shape(self.key_proj(hidden_states), -1, batch_size)
            value_states = self._shape(self.value_proj(hidden_states), -1, batch_size)

        if is_cross_attention:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse all cross-attention
            # key/value_states (first "if" case)
            # if encoder bi-directional self-attention `past_key_value` is always `None`
            cache_batch_size = (batch_size // self.num_beams)
            if past_key_value is None:
                cache_shape = (cache_batch_size, self.num_beams, self.num_attn_heads, -1, self.head_dim)
                key_states = key_states.view(cache_shape)[:, 0 : 1, :, :, :].contiguous()
                value_states = value_states.view(cache_shape)[:, 0 : 1, :, :, :].contiguous()
                past_key_value = (key_states, value_states)
                past_key_value = (key_states, value_states)

        # project states into the correct shape
        proj_shape = (batch_size * self.num_attn_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, batch_size).view(*proj_shape)
        
        if is_cross_attention:
            query_states = query_states.view(cache_batch_size, self.num_beams, self.num_attn_heads, tgt_len,
                       self.head_dim)   
            src_len = key_states.size(3)
            attn_weights = torch.einsum("bmhtd,bnhsd->bmhts", 
                                        query_states, key_states).reshape(-1, tgt_len, src_len)
            assert attn_weights.size() == (batch_size * self.num_attn_heads, tgt_len,
                                           src_len)
        else:
            key_states = key_states.view(*proj_shape)
            value_states = value_states.view(*proj_shape)
            src_len = key_states.size(1)
            attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))
            assert attn_weights.size() == (batch_size * self.num_attn_heads, tgt_len,
                                           src_len)

        assert attn_weights.size() == (
            batch_size * self.num_attn_heads,
            tgt_len,
            src_len,
        ), f"`attn_weights` should be of size {batch_size * self.num_attn_heads, tgt_len, src_len}, but is of size {attn_weights.shape}"

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if attention_mask is not None and attention_mask.dim() == 0:
            attention_mask = None
        assert attention_mask is None or attention_mask.size() == (
            self.num_attn_heads * batch_size,
            1,
            src_len,
        ), f"`attention_mask` should be `None` or of shape attention_mask.size() == {batch_size * self.num_attn_heads, 1, src_len}, but is {attention_mask.shape}"

        if attention_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights + attention_mask

        if output_attentions:
            # this operation is a bit awkward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to be reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(batch_size, self.num_attn_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(batch_size * self.num_attn_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if layer_head_mask is not None:
            assert layer_head_mask.size() == (
                self.num_attn_heads,
            ), f"Head mask for a single layer should be of size {(self.num_attn_heads,)}, but is {layer_head_mask.size()}"
            attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights.view(
                batch_size, self.num_attn_heads, tgt_len, src_len
            )
            attn_weights = attn_weights.view(batch_size * self.num_attn_heads, tgt_len, src_len)

            # apply head_mask also on attn_weights_reshaped which is used for n-gram attention inside the model
            attn_weights_reshaped = layer_head_mask.view(1, -1, 1, 1) * attn_weights_reshaped

        attn_probs = nn.functional.dropout(
            attn_weights,
            p=self.attention_dropout,
            training=self.training,
        )

        if is_cross_attention:
            attn_probs = attn_probs.view(cache_batch_size, self.num_beams, self.num_attn_heads, tgt_len, src_len)
            attn_output = torch.einsum("bmhts,bnhsd->bmhtd", attn_probs, value_states).reshape(-1, tgt_len, self.head_dim)
        else:
            attn_output = torch.bmm(attn_probs, value_states)
        assert attn_output.size() == (
            batch_size * self.num_attn_heads,
            tgt_len,
            self.head_dim,
        ), "`attn_output` should be of shape {batch_size * self.num_attn_heads, tgt_len, self.head_dim}, but is of shape {attn_output.size()}"

        attn_output = (
            attn_output.view(batch_size, self.num_attn_heads, tgt_len, self.head_dim)
            .transpose(1, 2)
            .reshape(batch_size, tgt_len, hidden_size)
        )

        attn_output = self.out_proj(attn_output)

        attn_output = nn.functional.dropout(attn_output, p=self.dropout, training=self.training)
        return attn_output, attn_weights_reshaped, past_key_value
        