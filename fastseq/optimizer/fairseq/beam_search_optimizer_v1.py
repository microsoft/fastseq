# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Apply the beam search optimizations to fairseq-v0.9.0"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from fairseq import utils
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
    def _reorder_encoder_out(self, encoder_out, new_order):
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

    def __init__(self,
                 embed_dim,
                 num_heads,
                 kdim=None,
                 vdim=None,
                 dropout=0.,
                 bias=True,
                 add_bias_kv=False,
                 add_zero_attn=False,
                 self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__(embed_dim, num_heads, kdim, vdim, dropout, bias,
                         add_bias_kv, add_zero_attn, self_attention,
                         encoder_decoder_attention)

        self.beam_size = 1
        self.tpu = False

    def apply_sparse_mask(
        self, attn_weights, tgt_len: int, src_len: int, bsz: int):
        return attn_weights

    def forward(
        self,
        query,
        key,
        value,
        key_padding_mask=None,
        incremental_state=None,
        need_weights=True,
        static_kv=False,
        attn_mask=None,
        before_softmax=False,
        need_head_weights=False,
    ):
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

        if (self.enable_torch_version and
            not self.onnx_trace and
            incremental_state is None and
            not static_kv):
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
                v_proj_weight=self.v_proj.weight)

        if incremental_state is not None:
            saved_state = self._get_input_buffer(incremental_state)
            if 'prev_key' in saved_state:
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
                key_padding_mask = torch.cat([
                    key_padding_mask,
                    key_padding_mask.new_zeros(key_padding_mask.size(0), 1)
                ],
                                             dim=1)

        if k is not None:
            kv_bsz = k.size(1)
            k = k.contiguous().view(-1, kv_bsz * self.num_heads,
                                    self.head_dim).transpose(0, 1)
        if v is not None:
            assert kv_bsz
            v = v.contiguous().view(-1, kv_bsz * self.num_heads,
                                    self.head_dim).transpose(0, 1)

        if saved_state is not None:
            # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
            if 'prev_key' in saved_state:
                saved_prev_key = saved_state["prev_key"]
                assert saved_prev_key is not None
                kv_bsz = saved_prev_key.size(0)
                prev_key = saved_prev_key.view(kv_bsz * self.num_heads, -1,
                                               self.head_dim)
                if static_kv:
                    k = prev_key
                else:
                    assert k is not None
                    k = torch.cat((prev_key, k), dim=1)
            if 'prev_value' in saved_state:
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
            key_padding_mask = self._append_prev_key_padding_mask(
                key_padding_mask=key_padding_mask,
                prev_key_padding_mask=prev_key_padding_mask,
                batch_size=kv_bsz,
                src_len=k.size(1),
                static_kv=static_kv,
            )

            saved_state['prev_key'] = k.view(kv_bsz, self.num_heads, -1,
                                             self.head_dim)
            saved_state['prev_value'] = v.view(kv_bsz, self.num_heads, -1,
                                               self.head_dim)
            saved_state['prev_key_padding_mask'] = key_padding_mask
            # In this branch incremental_state is never None
            assert incremental_state is not None
            self._set_input_buffer(incremental_state, saved_state)

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
                key_padding_mask = torch.cat([
                    key_padding_mask,
                    torch.zeros(key_padding_mask.size(0),
                                1).type_as(key_padding_mask)
                ],
                                             dim=1)

        if self.encoder_decoder_attention and bsz != kv_bsz:
            #query size (1, B*b*h, c_embed) => (B*h, b, c)
            q = q.view(tgt_len,-1, self.beam_size, self.num_heads,
                    self.head_dim).permute(1,3,2,0,4).contiguous(
                    ).view(kv_bsz*self.num_heads, self.beam_size, self.head_dim)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
            attn_weights = attn_weights.view(-1, tgt_len,
                    *attn_weights.size()[-1:])
        else:
            q = q.contiguous().view(tgt_len, bsz * self.num_heads,
                                self.head_dim).transpose(0, 1)
            attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(
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
            #attn_weights size (B*b*h/B*h*b, 1, S) => (B,h*b, S)
            attn_weights = attn_weights.view(kv_bsz, -1, src_len)
            if not self.tpu:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).to(
                        torch.bool), float("-inf"))
            else:
                #Not supported
                attn_weights = attn_weights.transpose(0, 2)
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask, float('-inf'))
                attn_weights = attn_weights.transpose(0, 2)
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len,
                                             src_len)

        if before_softmax:
            #attn_weights size (B*h*b, 1, S) => (B*b*h, 1, S)
            if self.encoder_decoder_attention and bsz != kv_bsz:
                attn_weights = attn_weights.view(kv_bsz,self.num_heads,
                self.beam_size, tgt_len, src_len).permute(0,2,1,3,4
                ).contiguous().view(-1, tgt_len, src_len)
            return attn_weights, v

        attn_weights_float = utils.softmax(attn_weights,
                                           dim=-1,
                                           onnx_trace=self.onnx_trace)
        attn_weights = attn_weights_float.type_as(attn_weights)
        attn_probs = F.dropout(attn_weights,
                               p=self.dropout,
                               training=self.training)
        assert v is not None

        if self.encoder_decoder_attention and bsz != kv_bsz:
            #attn_probs size (B*h*b, 1, S) => (B*h, b, S)
            attn_probs = attn_probs.view(-1, self.beam_size, src_len)
        attn = torch.bmm(attn_probs, v)

        if self.encoder_decoder_attention and bsz != kv_bsz:
            assert list(
            attn.size()) == [kv_bsz * self.num_heads,
                    self.beam_size, self.head_dim]
        else:
            assert list(
            attn.size()) == [bsz * self.num_heads,
                    tgt_len, self.head_dim]

        if self.encoder_decoder_attention and bsz != kv_bsz:
            #attn size (B*h, b, c) => (1, B*b, c_embed)
            attn = attn.view(kv_bsz, self.num_heads,
                self.beam_size,self.head_dim).permute(0, 2, 1, 3
                ).contiguous().view(tgt_len, bsz, embed_dim)
                #.view(tgt_len, -1, self.head_dim
        elif (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0,
                                  1).contiguous().view(tgt_len, bsz, embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
             #attn_weights size (B*h*b,1, S) => (h,B*b,1,S)
            if self.encoder_decoder_attention and bsz != kv_bsz:
                attn_weights =  attn_weights_float.view(kv_bsz, self.num_heads,
	        self.beam_size, tgt_len, src_len).permute(1,0,2,3,4).contiguous(
	        ).view(self.num_heads, bsz, tgt_len, src_len)
            else:
                attn_weights = attn_weights_float.view(bsz, self.num_heads,
                                                    tgt_len,
                                                    src_len).transpose(1, 0)
            if not need_head_weights:
                # average attention weights over heads
                attn_weights = attn_weights.mean(dim=0)
        else:
            attn_weights = None

        return attn, attn_weights

    def reorder_incremental_state(self, incremental_state, new_order):
        """Reorder buffered internal state (for incremental generation)."""
        input_buffer = self._get_input_buffer(incremental_state)
        if input_buffer is not None:
            for k in input_buffer.keys():
                input_buffer_k = input_buffer[k]
                if input_buffer[k] is not None:
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
            self._set_input_buffer(incremental_state, input_buffer)

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

    @torch.no_grad()
    def _generate(self,
                  model,
                  sample,
                  prefix_tokens=None,
                  bos_token=None,
                  **kwargs):
        if not self.retain_dropout:
            model.eval()

        # model.forward normally channels prev_output_tokens into the decoder
        # separately, but SequenceGenerator directly calls model.encoder
        encoder_input = {
            k: v
            for k, v in sample['net_input'].items()
            if k != 'prev_output_tokens'
        }

        src_tokens = encoder_input['src_tokens']
        src_lengths = (src_tokens.ne(self.eos)
                       & src_tokens.ne(self.pad)).long().sum(dim=1)
        input_size = src_tokens.size()
        # batch dimension goes first followed by source lengths
        bsz = input_size[0]
        src_len = input_size[1]
        beam_size = self.beam_size

        if self.match_source_len:
            max_len = src_lengths.max().item()
        else:
            max_len = min(
                int(self.max_len_a * src_len + self.max_len_b),
                # exclude the EOS marker
                model.max_decoder_positions() - 1,
            )

        # compute the encoder output for each beam
        encoder_outs = model.forward_encoder(encoder_input)
        new_order = torch.arange(bsz).view(-1, 1).repeat(1, beam_size).view(-1)
        new_order = new_order.to(src_tokens.device).long()
        encoder_outs = model.reorder_encoder_out(encoder_outs, new_order)

        # initialize buffers
        scores = src_tokens.new(bsz * beam_size, max_len + 1).float().fill_(0)
        scores_buf = scores.clone()
        tokens = src_tokens.new(bsz * beam_size,
                                max_len + 2).long().fill_(self.pad)
        tokens_buf = tokens.clone()
        tokens[:, 0] = self.eos if bos_token is None else bos_token
        attn, attn_buf = None, None

        # The blacklist indicates candidates that should be ignored.
        # For example, suppose we're sampling and have already finalized 2/5
        # samples. Then the blacklist would mark 2 positions as being ignored,
        # so that we only finalize the remaining 3 samples.
        blacklist = src_tokens.new_zeros(bsz, beam_size).eq(
            -1)  # forward and backward-compatible False mask

        # list of completed sentences
        finalized = [[] for i in range(bsz)]
        finished = [False for i in range(bsz)]
        num_remaining_sent = bsz

        # number of candidate hypos per step
        cand_size = 2 * beam_size  # 2 x beam size in case half are EOS

        # offset arrays for converting between different indexing schemes
        bbsz_offsets = (torch.arange(0, bsz) *
                        beam_size).unsqueeze(1).type_as(tokens)
        cand_offsets = torch.arange(0, cand_size).type_as(tokens)

        # helper function for allocating buffers on the fly
        buffers = {}

        def buffer(name, type_of=tokens):  # noqa
            if name not in buffers:
                buffers[name] = type_of.new()
            return buffers[name]

        def is_finished(sent, step, unfin_idx):
            """
            Check whether we've finished generation for a given sentence, by
            comparing the worst score among finalized hypotheses to the best
            possible score among unfinalized hypotheses.
            """
            assert len(finalized[sent]) <= beam_size
            if len(finalized[sent]) == beam_size or step == max_len:
                return True
            return False

        def finalize_hypos(step, bbsz_idx, eos_scores):
            """
            Finalize the given hypotheses at this step, while keeping the total
            number of finalized hypotheses per sentence <= beam_size.

            Note: the input must be in the desired finalization order, so that
            hypotheses that appear earlier in the input are preferred to those
            that appear later.

            Args:
                step: current time step
                bbsz_idx: A vector of indices in the range [0, bsz*beam_size),
                    indicating which hypotheses to finalize
                eos_scores: A vector of the same size as bbsz_idx containing
                    scores for each hypothesis
            """
            assert bbsz_idx.numel() == eos_scores.numel()

            # clone relevant token and attention tensors
            tokens_clone = tokens.index_select(0, bbsz_idx)
            tokens_clone = tokens_clone[:, 1:step +
                                        2]  # skip the first index, which is EOS
            assert not tokens_clone.eq(self.eos).any()
            tokens_clone[:, step] = self.eos
            attn_clone = attn.index_select(
                0, bbsz_idx)[:, :, 1:step + 2] if attn is not None else None

            # compute scores per token position
            pos_scores = scores.index_select(0, bbsz_idx)[:, :step + 1]
            pos_scores[:, step] = eos_scores
            # convert from cumulative to per-position scores
            pos_scores[:, 1:] = pos_scores[:, 1:] - pos_scores[:, :-1]

            # normalize sentence-level scores
            if self.normalize_scores:
                eos_scores /= (step + 1)**self.len_penalty

            cum_unfin = []
            prev = 0
            for f in finished:
                if f:
                    prev += 1
                else:
                    cum_unfin.append(prev)

            sents_seen = set()
            for i, (idx, score) in enumerate(
                zip(bbsz_idx.tolist(), eos_scores.tolist())):
                unfin_idx = idx // beam_size
                sent = unfin_idx + cum_unfin[unfin_idx]

                sents_seen.add((sent, unfin_idx))

                if self.match_source_len and step > src_lengths[unfin_idx]:
                    score = -math.inf

                def get_hypo():

                    if attn_clone is not None:
                        # remove padding tokens from attn scores
                        hypo_attn = attn_clone[i]
                    else:
                        hypo_attn = None

                    return {
                        'tokens': tokens_clone[i],
                        'score': score,
                        'attention': hypo_attn,  # src_len x tgt_len
                        'alignment': None,
                        'positional_scores': pos_scores[i],
                    }

                if len(finalized[sent]) < beam_size:
                    finalized[sent].append(get_hypo())

            newly_finished = []
            for sent, unfin_idx in sents_seen:
                # check termination conditions for this sentence
                if not finished[sent] and is_finished(sent, step, unfin_idx):
                    finished[sent] = True
                    newly_finished.append(unfin_idx)
            return newly_finished

        reorder_state = None
        batch_idxs = None
        for step in range(max_len + 1):  # one extra step for EOS marker
            # reorder decoder internal states based on the prev choice of beams
            if reorder_state is not None:
                if batch_idxs is not None:
                    # update beam indices to take into account removed sentences
                    corr = batch_idxs - torch.arange(
                        batch_idxs.numel()).type_as(batch_idxs)
                    reorder_state.view(-1, beam_size).add_(
                        corr.unsqueeze(-1) * beam_size)
                model.reorder_incremental_state(reorder_state)
                encoder_outs = model.reorder_encoder_out(
                    encoder_outs, reorder_state)

            lprobs, avg_attn_scores = model.forward_decoder(
                tokens[:, :step + 1],
                encoder_outs,
                temperature=self.temperature,
            )

            lprobs[:, self.pad] = -math.inf  # never select pad
            lprobs[:, self.unk] -= self.unk_penalty  # apply unk penalty

            # handle max length constraint
            if step >= max_len:
                lprobs[:, :self.eos] = -math.inf
                lprobs[:, self.eos + 1:] = -math.inf

            # handle prefix tokens (possibly with different lengths)
            if prefix_tokens is not None and step < prefix_tokens.size(
                1) and step < max_len:
                prefix_toks = prefix_tokens[:, step].unsqueeze(-1).repeat(
                    1, beam_size).view(-1)
                prefix_lprobs = lprobs.gather(-1, prefix_toks.unsqueeze(-1))
                prefix_mask = prefix_toks.ne(self.pad)
                lprobs[prefix_mask] = -math.inf
                lprobs[prefix_mask] = lprobs[prefix_mask].scatter_(
                    -1, prefix_toks[prefix_mask].unsqueeze(-1),
                    prefix_lprobs[prefix_mask])
                # if prefix includes eos, then we should make sure tokens and
                # scores are the same across all beams
                eos_mask = prefix_toks.eq(self.eos)
                if eos_mask.any():
                    # validate that the first beam matches the prefix
                    first_beam = tokens[eos_mask].view(
                        -1, beam_size, tokens.size(-1))[:, 0, 1:step + 1]
                    eos_mask_batch_dim = eos_mask.view(-1, beam_size)[:, 0]
                    target_prefix = prefix_tokens[eos_mask_batch_dim][:, :step]
                    assert (first_beam == target_prefix).all()

                    def replicate_first_beam(tensor, mask):
                        tensor = tensor.view(-1, beam_size, tensor.size(-1))
                        tensor[mask] = tensor[mask][:, :1, :]
                        return tensor.view(-1, tensor.size(-1))

                    # copy tokens, scores and lprobs from the first beam to all beams
                    tokens = replicate_first_beam(tokens, eos_mask_batch_dim)
                    scores = replicate_first_beam(scores, eos_mask_batch_dim)
                    lprobs = replicate_first_beam(lprobs, eos_mask_batch_dim)
            elif step < self.min_len:
                # minimum length constraint (does not apply if using prefix_tokens)
                lprobs[:, self.eos] = -math.inf

            if self.no_repeat_ngram_size > 0:
                # for each beam and batch sentence, generate a list of previous ngrams
                banned_list = [[] for bbsz_idx in range(bsz * beam_size)]
                cpu_tokens = tokens.cpu()[:, :step + 1].numpy()
                check_start_pos = step + 2 - self.no_repeat_ngram_size
                for bbsz_idx in range(bsz * beam_size):
                    for i in range(check_start_pos):
                        is_banned = True
                        for k in range(self.no_repeat_ngram_size - 1):
                            if cpu_tokens[bbsz_idx, i + k] != cpu_tokens[
                                bbsz_idx, check_start_pos + k]:
                                is_banned = False
                                break
                        if is_banned:
                            banned_list[bbsz_idx].append(
                                cpu_tokens[bbsz_idx,
                                           i + self.no_repeat_ngram_size - 1])

            # Record attention scores
            if avg_attn_scores is not None:
                if attn is None:
                    attn = scores.new(bsz * beam_size, src_tokens.size(1),
                                      max_len + 2)
                    attn_buf = attn.clone()
                attn[:, :, step + 1].copy_(avg_attn_scores)

            scores = scores.type_as(lprobs)
            scores_buf = scores_buf.type_as(lprobs)
            eos_bbsz_idx = buffer('eos_bbsz_idx')
            eos_scores = buffer('eos_scores', type_of=scores)

            self.search.set_src_lengths(src_lengths)

            if self.no_repeat_ngram_size > 0:

                def calculate_banned_tokens(bbsz_idx):
                    # before decoding the next token, prevent decoding of ngrams that have already appeared
                    banned_tokens_per_sample = [
                        (bbsz_idx, t) for t in banned_list[bbsz_idx]
                    ]
                    return banned_tokens_per_sample

                banned_tokens = []
                if step + 2 - self.no_repeat_ngram_size >= 0:
                    for bbsz_idx in range(bsz * beam_size):
                        banned_tokens.extend(calculate_banned_tokens(bbsz_idx))

                if banned_tokens:
                    banned_tokens = torch.LongTensor(banned_tokens)
                    lprobs.index_put_(
                        tuple(banned_tokens.t()),
                        lprobs.new_tensor([-math.inf] * len(banned_tokens)))

            cand_scores, cand_indices, cand_beams = self.search.step(
                step,
                lprobs.view(bsz, -1, self.vocab_size),
                scores.view(bsz, beam_size, -1)[:, :, :step],
            )

            # cand_bbsz_idx contains beam indices for the top candidate
            # hypotheses, with a range of values: [0, bsz*beam_size),
            # and dimensions: [bsz, cand_size]
            cand_bbsz_idx = cand_beams.add(bbsz_offsets)

            # finalize hypotheses that end in eos, except for blacklisted ones
            # or candidates with a score of -inf
            eos_mask = cand_indices.eq(self.eos) & cand_scores.ne(-math.inf)
            eos_mask[:, :beam_size][blacklist] = 0

            # only consider eos when it's among the top beam_size indices
            torch.masked_select(
                cand_bbsz_idx[:, :beam_size],
                mask=eos_mask[:, :beam_size],
                out=eos_bbsz_idx,
            )

            finalized_sents = set()
            if eos_bbsz_idx.numel() > 0:
                torch.masked_select(
                    cand_scores[:, :beam_size],
                    mask=eos_mask[:, :beam_size],
                    out=eos_scores,
                )
                finalized_sents = finalize_hypos(step, eos_bbsz_idx,
                                                 eos_scores)
                num_remaining_sent -= len(finalized_sents)

            assert num_remaining_sent >= 0
            if num_remaining_sent == 0:
                break
            assert step < max_len

            if len(finalized_sents) > 0:
                new_bsz = bsz - len(finalized_sents)

                # construct batch_idxs which holds indices of batches to keep for the next pass
                batch_mask = cand_indices.new_ones(bsz)
                batch_mask[cand_indices.new(finalized_sents)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)

                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_bsz, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                if prefix_tokens is not None:
                    prefix_tokens = prefix_tokens[batch_idxs]
                src_lengths = src_lengths[batch_idxs]
                blacklist = blacklist[batch_idxs]

                scores = scores.view(bsz, -1)[batch_idxs].view(
                    new_bsz * beam_size, -1)
                scores_buf.resize_as_(scores)
                tokens = tokens.view(bsz, -1)[batch_idxs].view(
                    new_bsz * beam_size, -1)
                tokens_buf.resize_as_(tokens)
                if attn is not None:
                    attn = attn.view(bsz, -1)[batch_idxs].view(
                        new_bsz * beam_size, attn.size(1), -1)
                    attn_buf.resize_as_(attn)
                bsz = new_bsz
            else:
                batch_idxs = None

            # Set active_mask so that values > cand_size indicate eos or
            # blacklisted hypos and values < cand_size indicate candidate
            # active hypos. After this, the min values per row are the top
            # candidate active hypos.
            active_mask = buffer('active_mask')
            eos_mask[:, :beam_size] |= blacklist
            torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[:eos_mask.size(1)],
                out=active_mask,
            )

            # get the top beam_size active hypotheses, which are just the hypos
            # with the smallest values in active_mask
            active_hypos, new_blacklist = buffer('active_hypos'), buffer(
                'new_blacklist')
            torch.topk(active_mask,
                       k=beam_size,
                       dim=1,
                       largest=False,
                       out=(new_blacklist, active_hypos))

            # update blacklist to ignore any finalized hypos
            blacklist = new_blacklist.ge(cand_size)[:, :beam_size]
            assert (~blacklist).any(dim=1).all()

            active_bbsz_idx = buffer('active_bbsz_idx')
            torch.gather(
                cand_bbsz_idx,
                dim=1,
                index=active_hypos,
                out=active_bbsz_idx,
            )
            active_scores = torch.gather(
                cand_scores,
                dim=1,
                index=active_hypos,
                out=scores[:, step].view(bsz, beam_size),
            )

            active_bbsz_idx = active_bbsz_idx.view(-1)
            active_scores = active_scores.view(-1)

            # copy tokens and scores for active hypotheses
            torch.index_select(
                tokens[:, :step + 1],
                dim=0,
                index=active_bbsz_idx,
                out=tokens_buf[:, :step + 1],
            )
            torch.gather(
                cand_indices,
                dim=1,
                index=active_hypos,
                out=tokens_buf.view(bsz, beam_size, -1)[:, :, step + 1],
            )
            if step > 0:
                torch.index_select(
                    scores[:, :step],
                    dim=0,
                    index=active_bbsz_idx,
                    out=scores_buf[:, :step],
                )
            torch.gather(
                cand_scores,
                dim=1,
                index=active_hypos,
                out=scores_buf.view(bsz, beam_size, -1)[:, :, step],
            )

            # copy attention for active hypotheses
            if attn is not None:
                torch.index_select(
                    attn[:, :, :step + 2],
                    dim=0,
                    index=active_bbsz_idx,
                    out=attn_buf[:, :, :step + 2],
                )

            # swap buffers
            tokens, tokens_buf = tokens_buf, tokens
            scores, scores_buf = scores_buf, scores
            if attn is not None:
                attn, attn_buf = attn_buf, attn

            # reorder incremental state in decoder
            reorder_state = active_bbsz_idx

        # sort by score descending
        for sent_id, _ in enumerate(finalized):
            finalized[sent_id] = sorted(finalized[sent_id],
                                        key=lambda r: r['score'],
                                        reverse=True)
        return finalized
