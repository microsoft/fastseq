# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimization for GPT2 model"""

import torch
import torch.nn as nn

from transformers.modeling_gpt2 import Attention, GPT2Model

from fastseq.logging import get_logger
from fastseq.utils.api_decorator import replace

logger = get_logger(__name__)


@replace(Attention)
class AttentionV2(Attention):
    def __init__(self, nx, n_ctx, config, scale=False, num_beams=1):
        super().__init__(nx=nx, n_ctx=n_ctx, config=config, scale=scale)

        self.cache_input_key = None
        self.cache_input_value = None
        self.cache_input_len = -1
        self.num_beams = num_beams

    def _attn(self, q, k, v, attention_mask=None, head_mask=None,
              output_attentions=False):
        w1 = torch.einsum(
            "bmhtd,bnhsd->bmhts",
            q.view((q.size(0) // self.num_beams, self.num_beams) + q.shape[1:]),
            self.cache_input_key)
        w1 = w1.reshape((-1,) + w1.shape[2:])
        w2 = torch.matmul(q, k)
        w = torch.cat([w1, w2], dim=-1)
        if self.scale:
            w = w / (float(v.size(-1)) ** 0.5)
        nd, ns = w.size(-2), w.size(-1)
        mask = self.bias[:, :, ns - nd : ns, :ns]
        w = torch.where(mask.bool(), w, self.masked_bias.to(w.dtype))

        if attention_mask is not None:
            # Apply the attention mask
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)

        # Mask heads if we want to
        if head_mask is not None:
            w = w * head_mask

        split_weights = w.split(self.cache_input_len, dim=-1)
        w1 = split_weights[0]
        w1 = w1.view(
            (w1.size(0)//self.num_beams, self.num_beams) + w1.shape[1:])
        attn = torch.einsum(
            "bmhtd,bnhds->bmhts",
            w1,
            self.cache_input_value)
        attn = attn.reshape((-1,) + attn.shape[2:])
        if len(split_weights) == 2:
            attn += torch.matmul(split_weights[1], v)
        outputs = [attn]
        if output_attentions:
            outputs.append(w)

        return outputs

    def forward(self, x, layer_past=None, attention_mask=None, head_mask=None,
                use_cache=False, output_attentions=False):
        x = self.c_attn(x)
        query, key, value = x.split(self.split_size, dim=2)
        query = self.split_heads(query)
        key = self.split_heads(key, k=True)
        value = self.split_heads(value)
        if layer_past is not None:
            past_key = layer_past[0].transpose(-2, -1) # transpose back cf below
            past_value = layer_past[1]
            key = torch.cat((past_key, key), dim=-1)
            value = torch.cat((past_value, value), dim=-2)

        if use_cache is True:
            if layer_past is None:
                if self.cache_input_key is not None:
                    logger.warning(
                        "The previous cached key and value in GPT2 "
                        "self-attention layer have been updated. If this is not"
                        "on purpose, please add past/layer_past parameter when"
                        " call this model or self-attention layer.")
                self.cache_input_key = key.transpose(-2, -1)
                self.cache_input_value = value

                # remove the duplicated dimensions
                cache_shape = self.cache_input_key.shape
                cache_shape = (
                    cache_shape[0]//self.num_beams, self.num_beams,
                    ) + cache_shape[1:]
                self.cache_input_key = self.cache_input_key.view(
                    cache_shape)[:,:1,].contiguous()
                self.cache_input_value = self.cache_input_value.view(
                    cache_shape)[:,:1,].contiguous()
                self.cache_input_len = self.cache_input_key.size(-2)

                key = key[:, :, :, self.cache_input_len:]
                value = value[:, :, self.cache_input_len:, :]
            # transpose to have same shapes for stacking
            present = torch.stack((key.transpose(-2, -1), value))
        else:
            present = (None,)

        attn_outputs = self._attn(
            query, key, value, attention_mask, head_mask, output_attentions)
        a = attn_outputs[0]

        a = self.merge_heads(a)
        a = self.c_proj(a)
        a = self.resid_dropout(a)

        outputs = [a, present] + attn_outputs[1:]
        return outputs  # a, present, (attentions)

@replace(GPT2Model)
class GPT2ModelV2(GPT2Model):
    def forward(
        self,
        input_ids=None,
        past=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.GPT2Config`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the last layer of the model.
            If `past` is used only the last hidden-state of the sequences of shape :obj:`(batch_size, 1, hidden_size)` is output.
        past (:obj:`List[torch.FloatTensor]` of length :obj:`config.n_layers` with each tensor of shape :obj:`(2, batch_size, num_heads, sequence_length, embed_size_per_head)`):
            Contains pre-computed hidden-states (key and values in the attention blocks).
            Can be used (see `past` input) to speed up sequential decoding.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True``) is passed or when ``config.output_hidden_states=True``:
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """
        output_attentions = (output_attentions if output_attentions is not None
                             else self.config.output_attentions)
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None
            else self.config.output_hidden_states)
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and "
                             "inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError(
                "You have to specify either input_ids or inputs_embeds")

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])
        if position_ids is not None:
            position_ids = position_ids.view(-1, input_shape[-1])

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            # the beginning part of key and value is for the input sentence
            # where each beam is the same and will not be changed, so this part
            # of key and value is cached to avoid the recomputing. Here, the
            # shape of cached past needs to add the lenght of source input.
            past_length = past[0][0].size(-2) + self.h[0].attn.cache_input_len
        if position_ids is None:
            device = (input_ids.device if input_ids is not None
                      else inputs_embeds.device)
            position_ids = torch.arange(
                past_length,
                input_shape[-1] + past_length,
                dtype=torch.long,
                device=device)
            position_ids = position_ids.unsqueeze(0).view(-1, input_shape[-1])

        # Attention mask.
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            # We create a 3D attention mask from a 2D tensor mask.
            # Sizes are [batch_size, 1, 1, to_seq_length]
            # So we can broadcast to [batch_size, num_heads, from_seq_length,
            # to_seq_length]. this attention mask is more simple than the
            # triangular masking of causal attention used in OpenAI GPT, we just
            # need to prepare the broadcast dimension here.
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

            # Since attention_mask is 1.0 for positions we want to attend and
            # 0.0 for masked positions, this operation will create a tensor
            # which is 0.0 for positions we want to attend and -10000.0 for
            # masked positions. Since we are adding it to the raw scores before
            # the softmax, this is effectively the same as removing these
            # entirely.
            attention_mask = attention_mask.to(
                dtype=next(self.parameters()).dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)
        position_embeds = self.wpe(position_ids)
        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
        else:
            token_type_embeds = 0
        hidden_states = inputs_embeds + position_embeds + token_type_embeds
        hidden_states = self.drop(hidden_states)

        output_shape = input_shape + (hidden_states.size(-1),)

        presents = ()
        all_attentions = []
        all_hidden_states = ()
        for i, (block, layer_past) in enumerate(zip(self.h, past)):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (
                    hidden_states.view(*output_shape),)

            outputs = block(
                hidden_states,
                layer_past=layer_past,
                attention_mask=attention_mask,
                head_mask=head_mask[i],
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            hidden_states, present = outputs[:2]
            if use_cache is True:
                presents = presents + (present,)

            if output_attentions:
                all_attentions.append(outputs[2])

        hidden_states = self.ln_f(hidden_states)

        hidden_states = hidden_states.view(*output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if use_cache is True:
            outputs = outputs + (presents,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            # let the number of heads free (-1) so we can extract attention
            # even after head pruning
            attention_output_shape = input_shape[:-1] + (
                -1,) + all_attentions[0].shape[-2:]
            all_attentions = tuple(
                t.view(*attention_output_shape) for t in all_attentions)
            outputs = outputs + (all_attentions,)
        # last hidden state, (presents), (all hidden_states), (attentions)
        return outputs
