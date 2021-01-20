# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import logging
import math
import os

import numpy as np
import torch
import torch.nn.functional as F
from fastseq.optimizer.transformers.beam_search_optimizer import GenerationMixinV2
from fastseq.models.unilm_hf.configuration_unilm import UnilmConfig
from fastseq.models.unilm_hf.utils_hf import \
    get_checkpoint_from_transformer_cache
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.modeling_bert import (BertConfig, BertEmbeddings,
                                        BertIntermediate, BertOutput,
                                        BertPooler, BertPreTrainingHeads,
                                        BertSelfOutput)

logger = logging.getLogger(__name__)
BertLayerNorm = torch.nn.LayerNorm

UNILM_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "unilm-base-cased":
    "https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased.bin",
    "unilm-large-cased":
    "https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased.bin",
    "unilm1-base-cased":
    "https://unilm.blob.core.windows.net/ckpt/unilm1-base-cased.bin",
    "unilm1-large-cased":
    "https://unilm.blob.core.windows.net/ckpt/unilm1-large-cased.bin",
    "unilm1.2-base-uncased":
    "https://unilm.blob.core.windows.net/ckpt/unilm1.2-base-uncased.bin",
    "cnndm-unilm-base-cased":
    "https://unilm.blob.core.windows.net/ckpt/cnndm.unilm1-base-cased.bin",
}


def _reorder_buffer(attn_cache, beam_idx):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None and "enc" not in k:
            attn_cache[k] = input_buffer_k.index_select(0, beam_idx)
    return attn_cache


def _get_new_tensor(tensor, batch_idx, beam_idx, beam_size):
    tsz = tensor.size()
    tensor = tensor.view(-1, beam_size, *tsz[1:])
    tensor = tensor[batch_idx].view(-1, *tsz[1:])[beam_idx]
    return tensor


def _reorder_buffer_v2(attn_cache, batch_idx, beam_idx, beam_size):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            if "enc" in k:
                attn_cache[k] = (input_buffer_k if batch_idx is None else
                                 input_buffer_k.index_select(0, batch_idx))
            else:
                attn_cache[k] = _get_new_tensor(input_buffer_k, batch_idx,
                                                beam_idx, beam_size)
    return attn_cache


class UnilmPreTrainedModel(PreTrainedModel):
    """An abstract class to handle weights initialization and
    a simple interface for dowloading and loading pretrained models.
    """

    config_class = UnilmConfig
    base_model_prefix = "unilm"
    pretrained_model_archive_map = {
        **UNILM_PRETRAINED_MODEL_ARCHIVE_MAP,
    }

    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0,
                                       std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls,
                        pretrained_model_name_or_path,
                        reuse_position_embedding=None,
                        *model_args,
                        **kwargs):
        pretrained_model_archive_map = cls.pretrained_model_archive_map
        if pretrained_model_name_or_path in pretrained_model_archive_map:
            state_dict = get_checkpoint_from_transformer_cache(
                archive_file=pretrained_model_archive_map[
                    pretrained_model_name_or_path],
                pretrained_model_name_or_path=pretrained_model_name_or_path,
                pretrained_model_archive_map=pretrained_model_archive_map,
                cache_dir=kwargs.get("cache_dir", None),
                force_download=kwargs.get("force_download", None),
                proxies=kwargs.get("proxies", None),
                resume_download=kwargs.get("resume_download", None),
            )
            kwargs["state_dict"] = state_dict
        elif os.path.isfile(pretrained_model_name_or_path):
            kwargs["state_dict"] = torch.load(pretrained_model_name_or_path,
                                              map_location="cpu")

        if kwargs["state_dict"] is None:
            logger.info("unilm does't support the model !")
            raise NotImplementedError()

        config = kwargs.pop("config", None)
        state_dict = kwargs.pop("state_dict", None)

        # initialize new position embeddings (From Microsoft/UniLM)
        if not isinstance(config, PretrainedConfig):
            config_path = (config if config is not None else
                           pretrained_model_name_or_path)
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                *model_args,
                cache_dir=kwargs.get("cache_dir", None),
                return_unused_kwargs=True,
                force_download=kwargs.get("force_download", None),
                resume_download=kwargs.get("resume_download", None),
                proxies=kwargs.get("proxies", None),
                local_files_only=kwargs.pop("local_files_only", False),
                **kwargs,
            )
        else:
            model_kwargs = kwargs

        kwargs["config"] = config

        _k = "bert.embeddings.position_embeddings.weight"
        if _k in state_dict:
            if config.max_position_embeddings > state_dict[_k].shape[0]:
                logger.info("Resize > position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(
                    torch.ones(size=(config.max_position_embeddings,
                                     state_dict[_k].shape[1])),
                    dtype=torch.float,
                )
                new_postion_embedding = nn.Parameter(
                    data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(
                    mean=0.0, std=config.initializer_range)
                max_range = (config.max_position_embeddings
                             if reuse_position_embedding else old_vocab_size)
                shift = 0
                while shift < max_range:
                    delta = min(old_vocab_size, max_range - shift)
                    new_postion_embedding.data[
                        shift:shift + delta, :] = state_dict[_k][:delta, :]
                    logger.info("  CP [%d ~ %d] into [%d ~ %d]  " %
                                (0, delta, shift, shift + delta))
                    shift += delta
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding
            elif config.max_position_embeddings < state_dict[_k].shape[0]:
                logger.info("Resize < position embeddings !")
                old_vocab_size = state_dict[_k].shape[0]
                new_postion_embedding = state_dict[_k].data.new_tensor(
                    torch.ones(size=(config.max_position_embeddings,
                                     state_dict[_k].shape[1])),
                    dtype=torch.float,
                )
                new_postion_embedding = nn.Parameter(
                    data=new_postion_embedding, requires_grad=True)
                new_postion_embedding.data.normal_(
                    mean=0.0, std=config.initializer_range)
                new_postion_embedding.data.copy_(
                    state_dict[_k][:config.max_position_embeddings, :])
                state_dict[_k] = new_postion_embedding.data
                del new_postion_embedding
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            pretrained_model_name_or_path = cls.pretrained_model_archive_map[
                pretrained_model_name_or_path]

        model = super().from_pretrained(pretrained_model_name_or_path,
                                        *model_args,
                                        state_dict=state_dict,
                                        **kwargs)
        model.load_state_dict(state_dict, False)
        return model


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" %
                (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                history_states=None):
        new_query_layer = self.query(hidden_states)
        new_key_layer = self.key(hidden_states)
        new_value_layer = self.value(hidden_states)

        prev_enc_key_layer = history_states.get("prev_enc_key_layer")
        prev_enc_value_layer = history_states.get("prev_enc_value_layer")
        prev_dec_key_layer = history_states.get("prev_dec_key_layer")
        prev_dec_value_layer = history_states.get("prev_dec_value_layer")

        query_layer = self.transpose_for_scores(new_query_layer)
        key_layer = self.transpose_for_scores(new_key_layer)
        value_layer = self.transpose_for_scores(new_value_layer)
        if prev_enc_key_layer is not None:
            enc_size = prev_enc_key_layer.size()
            enc_attention_scores = torch.einsum(
                "bxhtd,bhsd->bxhts",
                query_layer.view(enc_size[0], -1,
                                 *query_layer.size()[1:]),
                prev_enc_key_layer,
            )
            enc_attention_scores = enc_attention_scores.reshape(
                -1,
                *enc_attention_scores.size()[2:])
            if prev_dec_key_layer is not None:
                key_layer = torch.cat((prev_dec_key_layer, key_layer), dim=2)
                value_layer = torch.cat((prev_dec_value_layer, value_layer),
                                        dim=2)
            dec_attention_scores = torch.matmul(query_layer,
                                                key_layer.transpose(-1, -2))
            enc_attention_scores = enc_attention_scores / math.sqrt(
                self.attention_head_size)
            dec_attention_scores = dec_attention_scores / math.sqrt(
                self.attention_head_size)
            attention_scores = torch.cat(
                (enc_attention_scores, dec_attention_scores), dim=-1)
            attention_scores = attention_scores + attention_mask
            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            enc_attention_probs = attention_probs[:, :, :, :enc_size[2]]
            dec_attention_probs = attention_probs[:, :, :, enc_size[2]:]
            enc_attention_probs = enc_attention_probs.to(
                prev_enc_value_layer.dtype)
            enc_context_layer = torch.einsum(
                "bxhtd,bhds->bxhts",
                enc_attention_probs.view(enc_size[0], -1,
                                         *enc_attention_probs.size()[1:]),
                prev_enc_value_layer,
            )
            enc_context_layer = enc_context_layer.reshape(
                -1,
                *enc_context_layer.size()[2:])
            dec_context_layer = torch.matmul(dec_attention_probs, value_layer)
            context_layer = enc_context_layer + dec_context_layer

        else:
            attention_scores = torch.matmul(query_layer,
                                            key_layer.transpose(-1, -2))
            attention_scores = attention_scores / math.sqrt(
                self.attention_head_size)
            attention_scores = attention_scores + attention_mask

            attention_probs = nn.Softmax(dim=-1)(attention_scores)
            attention_probs = self.dropout(attention_probs)

            if head_mask is not None:
                attention_probs = attention_probs * head_mask
            context_layer = torch.matmul(attention_probs, value_layer)

        if history_states is None or len(history_states) == 0:
            history_states.update(
                dict({
                    "prev_enc_key_layer": key_layer,
                    "prev_enc_value_layer": value_layer,
                }))
        else:
            history_states.update(
                dict({
                    "prev_enc_key_layer": prev_enc_key_layer,
                    "prev_enc_value_layer": prev_enc_value_layer,
                    "prev_dec_key_layer": key_layer[:, :, :-1, :],
                    "prev_dec_value_layer": value_layer[:, :, :-1, :],
                }))

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = ((context_layer,
                    attention_probs) if self.output_attentions else
                   (context_layer, ))
        return outputs


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads,
                          self.self.attention_head_size)
        heads = (set(heads) - self.pruned_heads
                 )  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(
            heads)
        self.self.all_head_size = (self.self.attention_head_size *
                                   self.self.num_attention_heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(self,
                input_tensor,
                attention_mask,
                head_mask=None,
                history_states=None):
        self_outputs = self.self(input_tensor,
                                 attention_mask,
                                 head_mask,
                                 history_states=history_states)
        attention_output = self.output(self_outputs[0], input_tensor)
        outputs = (attention_output,
                   ) + self_outputs[1:]  # add attentions if we output them
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                history_states=None):
        attention_outputs = self.attention(hidden_states,
                                           attention_mask,
                                           head_mask,
                                           history_states=history_states)
        attention_output = attention_outputs[0]
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output, ) + attention_outputs[
            1:]  # add attentions if we output them
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)])

    def forward(self,
                hidden_states,
                attention_mask,
                head_mask=None,
                history_states=None):
        all_hidden_states = ()
        all_attentions = ()

        for i, layer_module in enumerate(self.layer):
            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                head_mask[i],
                history_states=history_states[i],
            )

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states, )

            hidden_states = layer_outputs[0]
            if self.output_attentions:
                all_attentions = all_attentions + (layer_outputs[1], )

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states, )

        outputs = (hidden_states, )
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states, )
        if self.output_attentions:
            outputs = outputs + (all_attentions, )
        return outputs  # last-layer hidden state, (all hidden states), (all attentions), (all encoder layers)


class UnilmModel(UnilmPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings,
                                                      new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """Prunes heads of the model.
        heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
        See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids,
        token_type_ids=None,
        attention_mask=None,
        position_ids=None,
        head_mask=None,
        history_states=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = (head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
                    -1).unsqueeze(-1))
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1,
                                             -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                             )  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters(
            )).dtype)  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        if history_states is None:
            history_states = [
                dict().copy() for _ in range(self.config.num_hidden_layers)
            ]
        embedding_output = self.embeddings(input_ids,
                                           position_ids=position_ids,
                                           token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            history_states=history_states,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = ((
            sequence_output,
            pooled_output,
        ) + encoder_outputs[1:] + (history_states, )
                   )  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class UnilmForSeq2Seq(UnilmPreTrainedModel, GenerationMixinV2):
    def __init__(self, config):
        super().__init__(config)
        config.output_hidden_states = True
        self.config = config
        self.bert = UnilmModel(config)
        self.cls = BertPreTrainingHeads(config)

    def get_input_embeddings(self):
        return self.bert.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.bert.embeddings.word_embeddings.weight = value

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def _output_past(self, outputs):
        if len(outputs) == 0 or self.training:
            return False
        return True

    def forward(self, src_in, **kwargs):
        if self.training:
            # code for training
            return

        dec_token, token_mask, pos_ids = src_in
        dec_token = torch.cat([dec_token, self.dec_mask_token], 1)
        dec_len = dec_token.size(1)
        dec_token = dec_token[:, -2:]
        dec_mask = token_mask[:, dec_len -
                              2:dec_len, :self.src_state["src_len"] + dec_len]
        dec_pos = pos_ids[:, dec_len - 2:dec_len]
        history_states = kwargs["history_states"]

        outputs = self.bert(
            dec_token,
            self.dec_seg[:, dec_len - 2:dec_len],
            dec_mask,
            dec_pos,
            history_states=history_states,
        )
        output, _ = self.cls(outputs[0],
                             outputs[1])  # Pick the last step: (bh * bm) * d_h
        state4cache = [pos_ids, token_mask] + outputs[3]
        return output, state4cache

    @staticmethod
    def _reorder_cache(past, beam_idx):
        pos_ids, token_mask, history_states = past[0], past[1], past[2:]
        reordered_past = []
        for layer_past in history_states:
            reordered_past.append(_reorder_buffer(layer_past, beam_idx))
        newpast = [pos_ids, token_mask] + reordered_past
        return newpast

    def _reorder_cache_v2(self, past, batch_idx, beam_idx, num_beams):
        pos_ids, token_mask, history_states = past[0], past[1], past[2:]
        reordered_past = []
        for layer_past in history_states:
            reordered_past.append(
                _reorder_buffer_v2(layer_past, batch_idx, beam_idx, num_beams))
        pos_ids = _get_new_tensor(pos_ids, batch_idx, beam_idx, num_beams)
        token_mask = _get_new_tensor(token_mask, batch_idx, beam_idx,
                                     num_beams)
        self.dec_mask_token = _get_new_tensor(self.dec_mask_token, batch_idx,
                                              beam_idx, num_beams)
        self.dec_seg = _get_new_tensor(self.dec_seg, batch_idx, beam_idx,
                                       num_beams)
        newpast = [pos_ids, token_mask] + reordered_past
        return newpast

    def prepare_inputs_for_generation(self, token_ids, past=None, **kwargs):
        if past is None:
            active_batch_size, _ = token_ids.size()
            src_token, src_seg, src_pos, src_mask = (
                self.src_state["src_token"],
                self.src_state["src_seg"],
                self.src_state["src_pos"],
                self.src_state["src_mask"],
            )
            src_len = self.src_state["src_len"]
            outputs = self.bert(
                src_token[:, :src_len],
                src_seg[:, :src_len],
                src_mask[:, :src_len, :src_len],
                src_pos[:, :src_len],
            )
            token_pos = src_pos.repeat(1, self.num_beams).view(
                active_batch_size, src_pos.size(1))
            token_pos = token_pos[:, src_len:]
            token_mask = (src_mask.unsqueeze(1).repeat(1, self.num_beams, 1,
                                                       1).view(
                                                           active_batch_size,
                                                           src_mask.size(1),
                                                           src_mask.size(1)))
            token_mask = token_mask[:, src_len:, :]
            history_states = outputs[3]
        else:
            token_pos, token_mask, history_states = past[0], past[1], past[2:]

        ret = dict({
            "src_in": (token_ids, token_mask, token_pos),
            "history_states": history_states,
        })
        return ret

    # beam search
    def generate(self, input_ids, attention_mask, decoder_start_token_id,
                 no_repeat_ngram_size, *args, **kwargs):
        max_seq_length = kwargs.pop("dec_max_length", 48)
        min_seq_length = kwargs.pop("dec_min_length", 0)
        repetition_penalty = kwargs.pop("repetition_penalty", 1.0)
        no_repeat_ngram_size = kwargs.pop("no_repeat_ngram_size", 0)
        length_penalty = kwargs.pop("length_penalty", 1.0)
        self.num_beams = kwargs.pop("num_beams", 5)
        num_return_sequences = kwargs.pop("num_return_sequences", 1)
        src_token, src_mask1 = input_ids, attention_mask
        batch_size = src_token.size(0)
        src_len = src_token.size(1)
        total_seq_length = max_seq_length + src_len + 1
        src_mask = src_mask1[:, None, :].repeat(1, total_seq_length, 1)
        tgt_mask = torch.zeros(batch_size, total_seq_length,
                               max_seq_length + 1).to(src_mask)
        tri_mask = torch.ones(batch_size, total_seq_length,
                              max_seq_length + 1).to(src_mask)
        tgt_mask[:, src_len:, :] = torch.tril(tri_mask[:, src_len:, :])
        tgt_mask[:, :, 0] = 0
        src_mask = torch.cat((src_mask, tgt_mask), dim=-1)
        src_seg = torch.tensor([0] * src_len).to(src_token)
        src_seg = src_seg[None, :].repeat(batch_size, 1)
        src_pos0 = torch.ones(batch_size, max_seq_length + 1).to(input_ids)
        src_pos0[:, 0] = 0
        src_pos = torch.cat((input_ids, src_pos0.to(input_ids)), dim=-1).ne(0)
        src_pos = torch.cumsum(src_pos, dim=-1) - 1
        self.src_state = dict({
            "src_len": src_len,
            "src_token": src_token,
            "src_seg": src_seg,
            "src_mask": src_mask,
            "src_pos": src_pos,
        })
        dec_seg = [0] + [1] * max_seq_length
        self.dec_seg = (torch.tensor(
            dec_seg, dtype=torch.long,
            device=src_token.device).unsqueeze(0).repeat(
                src_token.size(0) * self.num_beams, 1))
        self.dec_mask_token = (torch.from_numpy(
            np.array([self.config.mask_token_id
                      ])).repeat([batch_size * self.num_beams
                                  ]).unsqueeze(-1).to(src_token.device))
        if decoder_start_token_id is not None:
            self.config.bos_token_id = decoder_start_token_id
        bos_token = (torch.from_numpy(np.array([self.config.bos_token_id
                                                ])).repeat([batch_size
                                                            ]).unsqueeze(-1))
        if torch.cuda.is_available():
            bos_token = bos_token.cuda()

        batch_hyp = super().generate(
            bos_token,
            max_length=max_seq_length - 1,
            min_length=min_seq_length,
            do_sample=False,
            num_beams=self.num_beams,
            no_repeat_ngram_size=no_repeat_ngram_size,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            bos_token_id=self.config.bos_token_id,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id,
            num_return_sequences=num_return_sequences,
        )

        batch_hyp = batch_hyp.reshape(batch_size, num_return_sequences, -1)
        batch_hyp = batch_hyp[:, 0, :]
        return batch_hyp
