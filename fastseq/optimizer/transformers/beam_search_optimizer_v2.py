# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import inspect
import logging
import math
import os
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import Tensor, device, dtype, nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from fastseq.ops.ngram_repeat_block import NGramRepeatBlock
from .beam_search_optimizer import GenerationMixinV2


class GenerationMixinV3(GenerationMixinV2):
    def calculate_banned_tokens_from_bad_words(
        self,
        tokens,
        bad_words_dict: Dict[str, List[int]],
        bbsz_idx: int,
    ):
        bad_indice = []
        tokens_list = tokens[bbsz_idx].tolist()
        for ngram_size in bad_words_dict["ngram_size"]:
            ngram_index = ",".join([str(x) for x in tokens_list[-(ngram_size - 1):]])
            bad_indice += bad_words_dict.get(ngram_index, torch.jit.annotate(List[int], []))
        return bad_indice

    def _no_bad_words(self, tokens, lprobs, bsz: int, beam_size: int, bad_words_dict=None):
        if bad_words_dict is None:
            return lprobs

        banned_tokens = [
            self.calculate_banned_tokens_from_bad_words(
                tokens, bad_words_dict, bbsz_idx
            )
            for bbsz_idx in range(bsz * beam_size)
        ]
        for bbsz_idx in range(bsz * beam_size):
            lprobs[bbsz_idx][
                torch.tensor(banned_tokens[bbsz_idx]).long()
            ] = torch.tensor(-math.inf, dtype=torch.float).to(lprobs)
        return lprobs

    def _beam_search_step(self, step, scores, beam_scores):
        batch_size, num_beams, vocab_size = scores.size()
        beam_scores = scores + beam_scores
        beam_scores = beam_scores.view(batch_size, num_beams * vocab_size)
        cand_scores, next_tokens = torch.topk(beam_scores, 2 * num_beams, dim=1, largest=True, sorted=True)
        cand_beams = next_tokens // vocab_size
        cand_indices = next_tokens.fmod(vocab_size)
        return cand_scores, cand_indices, cand_beams

    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        min_length,
        do_sample,
        early_stopping,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        no_repeat_ngram_size,
        bad_words_ids,
        pad_token_id,
        eos_token_id,
        batch_size,
        num_return_sequences,
        length_penalty,
        num_beams,
        vocab_size,
        encoder_outputs,
        attention_mask,
        use_cache,
        model_specific_kwargs,
    ):
        """Generate sequences for each example with beam search."""
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)
        past = encoder_outputs
        cand_size = 2 * num_beams
        bbsz_offsets = (torch.arange(0, batch_size) * num_beams).unsqueeze(1).type_as(input_ids)
        cand_offsets = torch.arange(0, cand_size).type_as(input_ids)
        cands_to_ignore = torch.zeros(batch_size, num_beams).to(input_ids).eq(-1)
        num_remaining_sent = batch_size
        finished = [False for _ in range(batch_size)]
        bad_words_dict = None
        single_bad_words = None
        if bad_words_ids is not None:
            single_bad_words = [ngram[0] for ngram in bad_words_ids if len(ngram) == 1]
            bad_words_ids = [ngram for ngram in bad_words_ids if len(ngram) > 1]
            if len(bad_words_ids) == 0:
                bad_words_ids = None
            else:
                bad_words_dict: Dict[str, List[int]] = torch.jit.annotate(Dict[str, List[int]], {})
                for ngram in bad_words_ids:
                    key = ",".join([str(x) for x in ngram[:-1]])
                    bad_words_dict[key] = bad_words_dict.get(
                        key, torch.jit.annotate(List[int], [])
                    ) + [ngram[-1]]
                    bad_words_dict["ngram_size"] = bad_words_dict.get("ngram_size", torch.jit.annotate(List[int], [])) + [len(ngram)]
                bad_words_dict["ngram_size"] = list(set(bad_words_dict.get("ngram_size", torch.jit.annotate(List[int], []))))

        finalized = torch.jit.annotate(
            List[List[Dict[str, Tensor]]],
            [torch.jit.annotate(List[Dict[str, Tensor]], []) for i in range(batch_size)],
        )
        self.no_repeat_ngram_op = NGramRepeatBlock()#.to('cuda', torch.float32)
        start_len = cur_len
        while cur_len < max_length + 1:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask, use_cache=use_cache, **model_specific_kwargs
            )
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]

            scores = F.log_softmax(next_token_logits, dim=-1)
            if repetition_penalty != 1.0:
                prev_token_ids = input_ids[:, start_len-1:].long()
                curr_scores = scores[torch.arange(prev_token_ids.size(0)), prev_token_ids.transpose(0, 1)].transpose(0, 1)
                rep_penalty = torch.zeros_like(prev_token_ids).fill_(repetition_penalty).to(curr_scores)
                rep_penalty = (curr_scores < 0).float() * rep_penalty + (curr_scores > 0).float() / rep_penalty
                scores.scatter_(-1, prev_token_ids, rep_penalty * curr_scores)

            if eos_token_id is not None and cur_len < min_length:
                scores[:, eos_token_id] = -float("inf")

            if eos_token_id is not None and cur_len >= max_length:
                scores[:, : eos_token_id] = -float("inf")
                scores[:, eos_token_id + 1 :] = -float("inf")

            if no_repeat_ngram_size > 0:
                scores = self.no_repeat_ngram_op(input_ids[:, start_len-1:], scores.float(), batch_size, cur_len-start_len, num_beams, no_repeat_ngram_size)

            if single_bad_words is not None:
                scores[:, single_bad_words] = -float("inf")

            if bad_words_ids is not None:
                scores = self._no_bad_words(input_ids[:, start_len-1:], scores, batch_size, num_beams, bad_words_dict)

            beam_scores = beam_scores[:, None].expand_as(scores)
            scores = scores.view(batch_size, num_beams, -1)
            beam_scores = beam_scores.view(batch_size, num_beams, -1)
            cand_scores, cand_indices, cand_beams = self._beam_search_step(cur_len - start_len, scores, beam_scores)

            cand_bbsz_idx = cand_beams.add(bbsz_offsets)
            eos_mask = cand_indices.eq(eos_token_id)
            eos_mask[:, :num_beams][cands_to_ignore] = torch.tensor(0).to(eos_mask)
            eos_bbsz_idx = torch.masked_select(cand_bbsz_idx[:, :num_beams], mask=eos_mask[:, :num_beams])
            finalized_sents: List[int] = []
            if eos_bbsz_idx.numel() > 0:
                eos_scores = torch.masked_select(cand_scores[:, :num_beams], mask=eos_mask[:, :num_beams])
                input_clone = input_ids.index_select(0, eos_bbsz_idx)
                cum_unfin: List[int] = []
                prev = 0
                for f in finished:
                    if f:
                        prev += 1
                    else:
                        cum_unfin.append(prev)
                sents_seen: Dict[str, Optional[Tensor]] = {}
                for i in range(eos_bbsz_idx.size(0)):
                    idx = eos_bbsz_idx[i]
                    score = eos_scores[i]
                    unfin_idx = idx // num_beams
                    sent = unfin_idx + cum_unfin[unfin_idx]
                    seen = str(sent.item()) + "_" + str(unfin_idx.item())
                    if seen not in sents_seen:
                        sents_seen[seen] = None
                    if len(finalized[sent]) < num_beams:
                        finalized[sent].append({
                            'tokens': input_clone[i],
                            'score': score / (cur_len - start_len + 1) ** length_penalty
                        })
                for seen in sents_seen.keys():
                    sent = int(float(seen.split("_")[0]))
                    unfin_idx = int(float(seen.split("_")[1]))
                    if not finished[sent] and self.is_finished(cur_len, unfin_idx, max_length, len(finalized[sent]), num_beams):
                        finished[sent] = True
                        finalized_sents.append(unfin_idx)
                num_remaining_sent -= len(finalized_sents)

            if num_remaining_sent == 0:
                break

            if len(finalized_sents) > 0:
                new_batch_size = batch_size - len(finalized_sents)
                batch_mask = torch.ones(batch_size).to(cand_indices)
                batch_mask[torch.tensor(finalized_sents).to(cand_indices)] = torch.tensor(0).to(batch_mask)
                batch_idxs = batch_mask.nonzero().squeeze(-1)
                eos_mask = eos_mask[batch_idxs]
                cand_beams = cand_beams[batch_idxs]
                bbsz_offsets.resize_(new_batch_size, 1)
                cand_bbsz_idx = cand_beams.add(bbsz_offsets)
                cand_scores = cand_scores[batch_idxs]
                cand_indices = cand_indices[batch_idxs]
                cands_to_ignore = cands_to_ignore[batch_idxs]

                input_ids = input_ids.view(batch_size, -1)[batch_idxs].view(new_batch_size * num_beams, -1)
                batch_size = new_batch_size
            else:
                batch_idxs = None

            eos_mask[:, :num_beams] = ~((~cands_to_ignore) & (~eos_mask[:, :num_beams]))
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * cand_size,
                cand_offsets[: eos_mask.size(1)],
            )
            new_cands_to_ignore, active_hypos = torch.topk(
                active_mask, k=num_beams, dim=1, largest=False
            )
            cands_to_ignore = new_cands_to_ignore.ge(cand_size)[:, :num_beams]
            active_bbsz_idx = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            active_scores = torch.gather(cand_scores, dim=1, index=active_hypos)
            beam_idxs = torch.gather(cand_bbsz_idx, dim=1, index=active_hypos)
            beam_idxs = beam_idxs.view(-1)
            active_scores = active_scores.view(-1)
            input_ids = torch.index_select(input_ids, dim=0, index=beam_idxs)
            new_token = torch.gather(cand_indices, dim=1, index=active_hypos).view(-1)
            beam_scores = torch.gather(cand_scores, dim=1, index=active_hypos)
            beam_scores = beam_scores.view(-1)
            input_ids = torch.cat([input_ids, new_token.unsqueeze(1)], dim=-1)

            if past is not None:
                past = self._reorder_cache_v3(past, batch_idxs, beam_idxs)
            cur_len += 1

        batch_size = len(finalized)
        output_batch_size = batch_size if do_sample else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 if do_sample else num_return_sequences
        decoded = torch.zeros((output_batch_size, max_length)).long().to(input_ids).fill_(pad_token_id)

        # retrieve best hypotheses
        for i, sent in enumerate(range(len(finalized))):
            BCList = [
                BeamContainer(elem["score"].item(), elem) for elem in finalized[sent]
            ]
            BCList.sort()
            BCList.reverse()
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = output_num_return_sequences_per_batch * i + j
                best_hyp = BCList[j].elem['tokens']
                decoded[effective_batch_idx, :best_hyp.size(0)].copy_(best_hyp)

        return decoded

    def is_finished(self, cur_len, unfin_idx, max_length, finalized_sent_len, num_beams):
        assert finalized_sent_len <= num_beams
        if finalized_sent_len == num_beams or cur_len == max_length:
            return True
        return False

class BeamContainer(object):
    def __init__(self, score: float, elem: Dict[str, Tensor]):
        self.score = score
        self.elem = elem

    def __lt__(self, other):
        # type: (BeamContainer) -> bool
        # Due to https://github.com/pytorch/pytorch/issues/20388,
        # this has to use old style type annotations
        # Match original behavior of sorted function when two scores are equal.
        return self.score <= other.score
