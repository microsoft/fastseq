# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimization for beam search related parts in Transformers."""

import logging
import numpy as np
from typing import Iterable, Optional
from itertools import accumulate

import torch
from torch import Tensor
from torch.nn import functional as F

from transformers.generation_utils import (BeamHypotheses, GenerationMixin,
                                           calc_banned_bad_words_ids,
                                           calc_banned_ngram_tokens,
                                           top_k_top_p_filtering)

from transformers.modeling_bart import BartForConditionalGeneration
from transformers.modeling_t5 import T5ForConditionalGeneration
from transformers.modeling_gpt2 import GPT2Model, GPT2LMHeadModel, GPT2DoubleHeadsModel
from fastseq.ops.ngram_repeat_block import NGramRepeatBlock

from fastseq.logging import get_logger
from fastseq.utils.api_decorator import replace

logger = get_logger(__name__, logging.INFO)

no_repeat_ngram_op = NGramRepeatBlock()

@replace(GenerationMixin)
class GenerationMixinV2(GenerationMixin):
    """
    A class contraining all of the functions supporting generation, to be used
    as a mixin in PreTrainedModel.
    """

    def _update_beam_size(self, num_beams):
        """
        Update num_beams in the decoder's self_attn and encoder_decoder_attn
        layers if they have been optimized.

        Different implementations of ConditionalGeneration class (e.g.
        T5ForConditionalGeneration and BartForConditionalGeneration) may have
        different attribute hierarchies and their self_attn and
        encoder_decoder_attn may have been optimized or not. As a result, this
        function need to handle different cases without breaking the program.
        """

         # Update num_beams for BART decoder attention layer
        if isinstance(self, BartForConditionalGeneration):
            for layer in self.model.decoder.layers:
                layer.encoder_attn.num_beams = num_beams
                layer.self_attn.num_beams = num_beams
            logger.debug(
                "num_beams has been updated to {}".format(num_beams))
            return

        # Update num_beams for T5 decoder attention layer
        if isinstance(self, T5ForConditionalGeneration):
            for block in self.decoder.block:
                block.layer[0].SelfAttention.num_beams = num_beams
                block.layer[1].EncDecAttention.num_beams = num_beams
            logger.debug(
                "num_beams has been updated to {}".format(num_beams))
            return

        if isinstance(self, GPT2Model):
            for block in self.h:
                block.attn.num_beams = num_beams
            logger.debug("num_beams has been updated to {}".format(num_beams))
            return

        if (isinstance(self, GPT2LMHeadModel) or
            isinstance(self, GPT2DoubleHeadsModel)):
            for block in self.transformer.h:
                block.attn.num_beams = num_beams
            logger.debug("num_beams has been updated to {}".format(num_beams))
            return

        logger.debug(
            "The num_beams optimization in self_attn and encoder_decoder_attn "
            "does not support {} yet.".format(self.__class__))


    @torch.no_grad()
    def generate(self,
                 input_ids: Optional[torch.LongTensor] = None,
                 max_length: Optional[int] = None,
                 min_length: Optional[int] = None,
                 do_sample: Optional[bool] = None,
                 early_stopping: Optional[bool] = None,
                 num_beams: Optional[int] = None,
                 temperature: Optional[float] = None,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 repetition_penalty: Optional[float] = None,
                 bad_words_ids: Optional[Iterable[int]] = None,
                 bos_token_id: Optional[int] = None,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None,
                 length_penalty: Optional[float] = None,
                 no_repeat_ngram_size: Optional[int] = None,
                 num_return_sequences: Optional[int] = None,
                 attention_mask: Optional[torch.LongTensor] = None,
                 decoder_start_token_id: Optional[int] = None,
                 use_cache: Optional[bool] = None,
                 **model_specific_kwargs) -> torch.LongTensor:
        r""" Generates sequences for models with a LM head. The method currently
        supports greedy decoding, beam-search decoding, sampling with
        temperature, sampling with top-k or nucleus sampling.

        Adapted in part from `Facebook's XLM beam search code`_.

        .. _`Facebook's XLM beam search code`:
           https://github.com/facebookresearch/XLM/blob/9e6f6814d17be4fe5b15f2e6c43eb2b2d76daeb4/src/model/transformer.py#L529


        Parameters:

            input_ids: (`optional`) `torch.LongTensor` of shape
                `(batch_size, sequence_length)`.
                The sequence used as a prompt for the generation. If `None` the
                method initializes it as an empty `torch.LongTensor` of shape
                `(1,)`.

            max_length: (`optional`) int
                The max length of the sequence to be generated.  Between
                `min_length` and infinity. Default to 20.

            min_length: (`optional`) int
                The min length of the sequence to be generated.  Between 0 and
                infinity. Default to 0.

            do_sample: (`optional`) bool
                If set to `False` greedy decoding is used. Otherwise sampling is
                used. Defaults to `False` as defined in
                `configuration_utils.PretrainedConfig`.

            early_stopping: (`optional`) bool
                if set to `True` beam search is stopped when at least
                `num_beams` sentences finished per batch. Defaults to `False` as
                defined in `configuration_utils.PretrainedConfig`.

            num_beams: (`optional`) int
                Number of beams for beam search. Must be between 1 and infinity.
                1 means no beam search. Default to 1.

            temperature: (`optional`) float
                The value used to module the next token probabilities. Must be
                strictly positive. Default to 1.0.

            top_k: (`optional`) int
                The number of highest probability vocabulary tokens to keep for
                top-k-filtering. Between 1 and infinity. Default to 50.

            top_p: (`optional`) float
                The cumulative probability of parameter highest probability
                vocabulary tokens to keep for nucleus sampling. Must be between
                0 and 1. Default to 1.

            repetition_penalty: (`optional`) float
                The parameter for repetition penalty. Between 1.0 and infinity.
                1.0 means no penalty. Default to 1.0.

            pad_token_id: (`optional`) int
                Padding token. Default to specicic model pad_token_id or None if
                it does not exist.

            bos_token_id: (`optional`) int
                BOS token. Defaults to `bos_token_id` as defined in the models
                config.

            eos_token_id: (`optional`) int
                EOS token. Defaults to `eos_token_id` as defined in the models
                config.

            length_penalty: (`optional`) float
                Exponential penalty to the length. Default to 1.

            no_repeat_ngram_size: (`optional`) int
                If set to int > 0, all ngrams of size `no_repeat_ngram_size` can
                only occur once.
            bad_words_ids: (`optional`) list of lists of int
                `bad_words_ids` contains tokens that are not allowed to be
                generated. In order to get the tokens of the words that should
                not appear in the generated text, use
                `tokenizer.encode(bad_word, add_prefix_space=True)`.

            num_return_sequences: (`optional`) int
                The number of independently computed returned sequences for each
                element in the batch. Default to 1.

            attention_mask (`optional`) obj: `torch.LongTensor` of same shape as
                `input_ids`. Mask to avoid performing attention on padding token
                indices. Mask values selected in ``[0, 1]``: ``1`` for tokens
                that are NOT MASKED, ``0`` for MASKED tokens. Defaults to `None`

                `What are attention masks? <../glossary.html#attention-mask>`__

            decoder_start_token_id=None: (`optional`) int
                If an encoder-decoder model starts decoding with a different
                token than BOS. Defaults to `None` and is changed to `BOS`
                later.

            use_cache: (`optional`) bool
                If `use_cache` is True, past key values are used to speed up
                decoding if applicable to model. Defaults to `True`.

            model_specific_kwargs: (`optional`) dict
                Additional model specific kwargs will be forwarded to the
                `forward` function of the model.

        Return:

            output: `torch.LongTensor` of shape
                `(batch_size * num_return_sequences, sequence_length)`
                sequence_length is either equal to max_length or shorter if all
                batches finished early due to the `eos_token_id`

        Examples::
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
            # Download model and configuration from S3 and cache.
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')
            outputs = model.generate(max_length=40)  # do greedy decoding
            print('Generated: {}'.format(
                tokenizer.decode(outputs[0], skip_special_tokens=True)))

            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained('openai-gpt')
            # Download model and configuration from S3 and cache.
            model = AutoModelWithLMHead.from_pretrained('openai-gpt')
            input_context = 'The dog'
            # encode input context
            input_ids = tokenizer.encode(input_context, return_tensors='pt')
            # generate 3 independent sequences using beam search decoding
            # (5 beams) with sampling from initial context 'The dog'
            outputs = model.generate(input_ids=input_ids,
                                     num_beams=5,
                                     num_return_sequences=3,
                                     temperature=1.5)
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(
                    i, tokenizer.decode(outputs[i], skip_special_tokens=True)))
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained('distilgpt2')
            # Download model and configuration from S3 and cache.
            model = AutoModelWithLMHead.from_pretrained('distilgpt2')
            input_context = 'The dog'
            # encode input context
            input_ids = tokenizer.encode(input_context, return_tensors='pt')
            # 3 generate sequences using by sampling
            outputs = model.generate(input_ids=input_ids,
                                     max_length=40,
                                     temperature=0.7,
                                     num_return_sequences=3)
            for i in range(3): #  3 output sequences were generated
                print('Generated {}: {}'.format(i, tokenizer.decode(
                    outputs[i], skip_special_tokens=True)))

            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained('ctrl')
            # Download model and configuration from S3 and cache.
            model = AutoModelWithLMHead.from_pretrained('ctrl')
            # "Legal" is one of the control codes for ctrl
            input_context = 'Legal My neighbor is'
            # encode input context
            input_ids = tokenizer.encode(input_context, return_tensors='pt')
            # generate sequences
            outputs = model.generate(input_ids=input_ids,
                                     max_length=50,
                                     temperature=0.7,
                                     repetition_penalty=1.2)
            print('Generated: {}'.format(
                tokenizer.decode(outputs[0], skip_special_tokens=True)))
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained('gpt2')
            # Download model and configuration from S3 and cache.
            model = AutoModelWithLMHead.from_pretrained('gpt2')
            # "Legal" is one of the control codes for ctrl
            input_context = 'My cute dog'
            bad_words_ids = [tokenizer.encode(bad_word, add_prefix_space=True)
                             for bad_word in ['idiot', 'stupid', 'shut up']]
            # encode input context
            input_ids = tokenizer.encode(input_context, return_tensors='pt')
            # generate sequences without allowing bad_words to be generated
            outputs = model.generate(input_ids=input_ids,
                                     max_length=100,
                                     do_sample=True,
                                     bad_words_ids=bad_words_ids)
        """

        # We cannot generate if the model does not have a LM head
        if self.get_output_embeddings() is None:
            raise AttributeError(
                "You tried to generate sequences with a model that does not"
                "have a LM Head. Please use another model class (e.g. "
                "`OpenAIGPTLMHeadModel`, `XLNetLMHeadModel`, `GPT2LMHeadModel`,"
                "`CTRLLMHeadModel`, `T5WithLMHeadModel`, `TransfoXLLMHeadModel`"
                ", `XLMWithLMHeadModel`, `BartForConditionalGeneration` )"
            )

        max_length = (max_length
                      if max_length is not None else self.config.max_length)
        min_length = (min_length
                      if min_length is not None else self.config.min_length)
        do_sample = (do_sample
                     if do_sample is not None else self.config.do_sample)
        early_stopping = (early_stopping
                          if early_stopping is not None else
                          self.config.early_stopping)
        use_cache = (use_cache
                     if use_cache is not None else self.config.use_cache)
        num_beams = (num_beams
                     if num_beams is not None else self.config.num_beams)
        temperature = (temperature
                       if temperature is not None else self.config.temperature)
        top_k = top_k if top_k is not None else self.config.top_k
        top_p = top_p if top_p is not None else self.config.top_p
        repetition_penalty = (repetition_penalty
                              if repetition_penalty is not None else
                              self.config.repetition_penalty)
        bos_token_id = (bos_token_id
                        if bos_token_id is not None else
                        self.config.bos_token_id)
        pad_token_id = (pad_token_id
                        if pad_token_id is not None else
                        self.config.pad_token_id)
        eos_token_id = (eos_token_id
                        if eos_token_id is not None else
                        self.config.eos_token_id)
        length_penalty = (length_penalty
                          if length_penalty is not None else
                          self.config.length_penalty)
        no_repeat_ngram_size = (no_repeat_ngram_size
                                if no_repeat_ngram_size is not None else
                                self.config.no_repeat_ngram_size)
        bad_words_ids = (bad_words_ids if bad_words_ids is not None else
                         self.config.bad_words_ids)
        num_return_sequences = (num_return_sequences
                                if num_return_sequences is not None else
                                self.config.num_return_sequences)
        decoder_start_token_id = (decoder_start_token_id
                                  if decoder_start_token_id is not None else
                                  self.config.decoder_start_token_id)

        if input_ids is not None:
            batch_size = input_ids.shape[0]  # overriden by the input batch_size
        else:
            batch_size = 1

        assert isinstance(
            max_length, int
        ) and max_length > 0, (
            "`max_length` should be a strictly positive integer.")
        assert isinstance(
            min_length, int
        ) and min_length >= 0, "`min_length` should be a positive integer."
        assert isinstance(do_sample, bool), "`do_sample` should be a boolean."
        assert isinstance(early_stopping,
                          bool), "`early_stopping` should be a boolean."
        assert isinstance(use_cache, bool), "`use_cache` should be a boolean."
        assert isinstance(
            num_beams, int
        ) and num_beams > 0, (
            "`num_beams` should be a strictly positive integer.")
        assert temperature > 0, "`temperature` should be strictly positive."
        assert isinstance(
            top_k, int) and top_k >= 0, "`top_k` should be a positive integer."
        assert 0 <= top_p <= 1, "`top_p` should be between 0 and 1."
        assert repetition_penalty >= 1.0, "`repetition_penalty` should be >= 1."
        assert input_ids is not None or (
            isinstance(bos_token_id, int) and bos_token_id >= 0
        ), ("If input_ids is not defined, `bos_token_id` should be a positive"
            "integer.")
        assert pad_token_id is None or (isinstance(pad_token_id, int) and (
            pad_token_id >= 0)), "`pad_token_id` should be a positive integer."
        assert (eos_token_id is None) or (isinstance(eos_token_id, int) and (
            eos_token_id >= 0)), "`eos_token_id` should be a positive integer."
        assert length_penalty > 0, (
            "`length_penalty` should be strictly positive.")
        assert (isinstance(no_repeat_ngram_size, int)
                and no_repeat_ngram_size >= 0
                ), "`no_repeat_ngram_size` should be a positive integer."
        assert (
            isinstance(num_return_sequences, int) and num_return_sequences > 0
        ), "`num_return_sequences` should be a strictly positive integer."
        assert (
            bad_words_ids is None or isinstance(bad_words_ids, list)
            and isinstance(bad_words_ids[0], list)
        ), ("`bad_words_ids` is either `None` or a list of lists of tokens that"
            "should not be generated")

        if input_ids is None:
            assert isinstance(bos_token_id, int) and bos_token_id >= 0, (
                "you should either supply a context to complete as `input_ids`"
                " input or a `bos_token_id` (integer >= 0) as a first token to "
                "start the generation."
            )
            input_ids = torch.full(
                (batch_size, 1),
                bos_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
        else:
            assert input_ids.dim(
            ) == 2, ("Input prompt should be of shape (batch_size, sequence "
                     "length).")

        # not allow to duplicate outputs when greedy decoding
        if do_sample is False:
            if num_beams == 1:
                # no_beam_search greedy generation conditions
                assert (
                    num_return_sequences == 1
                ), ("Greedy decoding will always produce the same output for "
                    "num_beams == 1 and num_return_sequences > 1. Please set "
                    "num_return_sequences = 1")

            else:
                # beam_search greedy generation conditions
                assert (
                    num_beams >= num_return_sequences
                ), ("Greedy beam search decoding cannot return more sequences "
                    "than it has beams. "
                    "Please set num_beams >= num_return_sequences")

        # create attention mask if necessary
        # TODO (PVP): this should later be handled by the forward fn() in each
        # model in the future see PR 3140
        if (attention_mask is None and
            pad_token_id is not None and
            pad_token_id in input_ids):
            attention_mask = input_ids.ne(pad_token_id).long()
        elif attention_mask is None:
            attention_mask = input_ids.new_ones(input_ids.shape)

        # set pad_token_id to eos_token_id if not set. Important that this is
        # done after attention_mask is created
        if pad_token_id is None and eos_token_id is not None:
            logger.warning(
                "Setting `pad_token_id` to {} (first `eos_token_id`) to "
                "generate sequence".format(eos_token_id))
            pad_token_id = eos_token_id

        # current position and vocab size
        if hasattr(self.config, "vocab_size"):
            vocab_size = self.config.vocab_size
        elif (self.config.is_encoder_decoder
              and hasattr(self.config, "decoder")
              and hasattr(self.config.decoder, "vocab_size")):
            vocab_size = self.config.decoder.vocab_size

        self._update_beam_size(num_beams)

        # set effective batch size and effective batch multiplier according to
        # do_sample
        if do_sample:
            effective_batch_size = batch_size * num_return_sequences
            effective_batch_mult = num_return_sequences
        else:
            effective_batch_size = batch_size
            effective_batch_mult = 1

        if self.config.is_encoder_decoder:
            if decoder_start_token_id is None:
                decoder_start_token_id = bos_token_id

            assert (
                decoder_start_token_id is not None
            ), ("decoder_start_token_id or bos_token_id has to be defined for "
                "encoder-decoder generation")
            assert hasattr(
                self, "get_encoder"
            ), "{} should have a 'get_encoder' function defined".format(self)
            assert callable(self.get_encoder), "{} should be a method".format(
                self.get_encoder)

            # get encoder and store encoder outputs
            encoder = self.get_encoder()
            encoder_outputs: tuple = encoder(input_ids,
                                             attention_mask=attention_mask)

        # Expand input ids if num_beams > 1 or num_return_sequences > 1
        if num_return_sequences > 1 or num_beams > 1:
            input_ids_len = input_ids.shape[-1]
            input_ids = input_ids.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len)
            attention_mask = attention_mask.unsqueeze(1).expand(
                batch_size, effective_batch_mult * num_beams, input_ids_len)

            input_ids = input_ids.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)
            attention_mask = attention_mask.contiguous().view(
                effective_batch_size * num_beams, input_ids_len
            )  # shape: (batch_size * num_return_sequences * num_beams, cur_len)

        if self.config.is_encoder_decoder:
            # create empty decoder_input_ids
            input_ids = torch.full(
                (effective_batch_size * num_beams, 1),
                decoder_start_token_id,
                dtype=torch.long,
                device=next(self.parameters()).device,
            )
            cur_len = 1

            assert (
                batch_size == encoder_outputs[0].shape[0]
            ), (f"expected encoder_outputs[0] to have 1st dimension bs="
                "{batch_size}, got {encoder_outputs[0].shape[0]} ")

            # expand batch_idx to assign correct encoder output for expanded
            # input_ids (due to num_beams > 1 and num_return_sequences > 1)
            expanded_batch_idxs = (
                torch.arange(batch_size)
                .view(-1, 1)
                .repeat(1, num_beams * effective_batch_mult)
                .view(-1)
                .to(input_ids.device)
            )
            # expand encoder_outputs
            encoder_outputs = (encoder_outputs[0].index_select(
                0, expanded_batch_idxs), *encoder_outputs[1:])

        else:
            encoder_outputs = None
            cur_len = input_ids.shape[-1]

        assert (
            cur_len < max_length
        ), (f"The context has {cur_len} number of tokens, but `max_length` is "
            f"only {max_length}. Please make sure that `max_length` is bigger "
            "than the number of tokens, by setting either "
            f"`generate(max_length=...,...)` or `config.max_length = ...`")

        if num_beams > 1:
            output = self._generate_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                early_stopping=early_stopping,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                num_return_sequences=num_return_sequences,
                length_penalty=length_penalty,
                num_beams=num_beams,
                vocab_size=vocab_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )
        else:
            output = self._generate_no_beam_search(
                input_ids,
                cur_len=cur_len,
                max_length=max_length,
                min_length=min_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                batch_size=effective_batch_size,
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                use_cache=use_cache,
                model_specific_kwargs=model_specific_kwargs,
            )

        return output

    def postprocess_next_token_scores(
        self,
        scores,
        input_ids,
        no_repeat_ngram_size,
        bad_words_ids,
        cur_len,
        min_length,
        max_length,
        eos_token_id,
        repetition_penalty,
        batch_size,
        num_beams,
    ):
        """Postprocess to update the next token scores"""
        # repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
        if repetition_penalty != 1.0:
            self.enforce_repetition_penalty_(
                scores,
                batch_size,
                num_beams,
                input_ids,
                repetition_penalty,
            )

        # set eos token prob to zero if min_length is not reached
        if eos_token_id is not None and cur_len < min_length:
            scores[:, eos_token_id] = -float("inf")

        def _update_scores(banned_tokens):
            banned_idx = [(bbsz_idx, banned_idx)
                          for bbsz_idx in range(len(banned_tokens))
                          for banned_idx in banned_tokens[bbsz_idx]]
            if banned_idx:
                banned_2d_idx = tuple(torch.LongTensor(list(zip(*banned_idx))))
                scores.index_put_(
                    banned_2d_idx,
                    scores.new_tensor(
                        [-float("inf") * banned_2d_idx[0].nelement()]))


        cpu_input_ids = input_ids.cpu()
        if no_repeat_ngram_size > 0:
            #custom op for Ngram repeat blocking
            if (input_ids.is_cuda and scores.is_cuda):
                scores = no_repeat_ngram_op(input_ids,scores.float(),
                        batch_size, cur_len-1, num_beams, no_repeat_ngram_size)
            else:
                num_batch_hypotheses = batch_size * num_beams
                banned_ngram_tokens = calc_banned_ngram_tokens_v2(
                    cpu_input_ids,
                    num_batch_hypotheses,
                    no_repeat_ngram_size,
                    cur_len,
                    self.config.pad_token_id)
                _update_scores(banned_ngram_tokens)

        if bad_words_ids is not None:
            # calculate a list of banned tokens according to bad words
            banned_bad_words_tokens = calc_banned_bad_words_ids(
                cpu_input_ids, bad_words_ids)

            _update_scores(banned_bad_words_tokens)

        return scores


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
        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty,
                            early_stopping=early_stopping)
            for _ in range(batch_size)
        ]

        # scores for each sentence in the beam
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float,
                                    device=input_ids.device)

        # for greedy decoding it is made sure that only tokens of the first
        # beam are considered to avoid sampling the exact same tokens three
        # times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)
        beams_offset = (torch.arange(0, batch_size) * num_beams).unsqueeze(1).type_as(input_ids)

        cand_size = 2 * num_beams
        cand_offsets = torch.arange(0, cand_size).type_as(input_ids)

        # cache compute states
        past = (encoder_outputs, None) if encoder_outputs is not None else None

        # done sentences
        done = [False for _ in range(batch_size)]
        """
        _reorder_cache_v2(past, batch_idxs, beam_idxs)
        Remove the finished batches during beam search, reorder_cache_v2 is used to support dynamic batch size.
        for cache tensors with shape like (batch_size, ~): tensor = tensor[batch_idxs]
        for cache tensors with shape like (batch_size * beam_size, ~): tensor = tensor[beam_idxs]
        """
        use_reorder_cache_v2 = hasattr(self, '_reorder_cache_v2')

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask,
                use_cache=use_cache, **model_specific_kwargs
            )
            # (batch_size * num_beams, cur_len, vocab_size)
            outputs = self(**model_inputs)
            # (batch_size * num_beams, vocab_size)
            next_token_logits = outputs[0][:, -1, :]

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]
            if self.config.is_encoder_decoder and do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better
                # solution
                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length
                )
            # (batch_size * num_beams, vocab_size)
            scores = F.log_softmax(next_token_logits, dim=-1)
            scores = self.postprocess_next_token_scores(
                scores=scores,
                input_ids=input_ids,
                no_repeat_ngram_size=no_repeat_ngram_size,
                bad_words_ids=bad_words_ids,
                cur_len=cur_len,
                min_length=min_length,
                max_length=max_length,
                eos_token_id=eos_token_id,
                repetition_penalty=repetition_penalty,
                batch_size=batch_size,
                num_beams=num_beams,
            )

            assert scores.shape == (batch_size * num_beams, vocab_size),\
                 "Shapes of scores: {} != {}".format(
                scores.shape, (batch_size * num_beams, vocab_size)
            )

            if do_sample:
                # (batch_size * num_beams, vocab_size)
                curr_scores = scores + beam_scores[:, None].expand_as(scores)
                # Temperature
                if temperature != 1.0:
                    curr_scores = curr_scores / temperature
                # Top-p/top-k filtering
                curr_scores = top_k_top_p_filtering(
                    curr_scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all
                # beam_idxs
                curr_scores = curr_scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare
                # tokens and match output of greedy beam search)
                probs = F.softmax(curr_scores, dim=-1)

                # (batch_size, num_beams * 2)
                next_tokens = torch.multinomial(
                    probs, num_samples=2 * num_beams)

                # Compute next scores
                # (batch_size, num_beams * 2)
                next_scores = torch.gather(curr_scores, -1, next_tokens)
                # sort the sampled vector to make sure that the first num_beams
                # samples are the best
                next_scores, next_scores_indices = torch.sort(
                                    next_scores, descending=True, dim=1)
                 # (batch_size, num_beams * 2)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)

            else:
                # (batch_size * num_beams, vocab_size)
                next_scores = scores + beam_scores[:, None].expand_as(scores)

                # re-organize to group the beam together (we are keeping top
                # hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores,
                                2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() == (
                batch_size, 2 * num_beams)
            # next batch beam content
            next_tokens_id = next_tokens % vocab_size
            next_beams_id = next_tokens // vocab_size
            effective_beam_id =  next_beams_id + beams_offset
            if eos_token_id is not None :
                eos_mask = next_tokens_id.eq(eos_token_id)
            else :
                eos_mask = torch.zeros_like(next_tokens_id).bool()
            eos_effective_idx = torch.masked_select(
                effective_beam_id[:, :num_beams], mask=eos_mask[:, :num_beams]
            )

            finished_batch_idxs = []
            if use_reorder_cache_v2 and eos_effective_idx.numel() > 0:
                eos_effective_scores = torch.masked_select(
                    next_scores[:, :num_beams], mask=eos_mask[:, :num_beams]
                )
                input_clone = input_ids.index_select(0, eos_effective_idx)
                unfin_offset = np.array(list(accumulate(done)))[np.array(done) == 0]
                for i in range(eos_effective_idx.size(0)):
                    eos_idx = eos_effective_idx[i]
                    eos_score = eos_effective_scores[i]
                    unfin_batch_idx = eos_idx // num_beams
                    batch_idx = unfin_batch_idx + unfin_offset[unfin_batch_idx]
                    if not done[batch_idx] :
                        generated_hyps[batch_idx.item()].add(
                            input_clone[i],
                            eos_score.item())
                    is_done = done[batch_idx]
                    done[batch_idx] = (
                        done[batch_idx] or
                        generated_hyps[batch_idx].is_done(
                        next_scores[unfin_batch_idx].max().item(), cur_len))
                    if is_done != done[batch_idx]:
                        finished_batch_idxs.append(unfin_batch_idx)

            if not use_reorder_cache_v2:
                eos_effective_scores = torch.masked_select(
                    next_scores[:, :num_beams], mask=eos_mask[:, :num_beams]
                )
                input_ids_cpu = input_ids.cpu()
                eos_effective_idx_cpu= eos_effective_idx.cpu()
                eos_effective_scores_cpu = eos_effective_scores.cpu()
                for i in range (0, eos_effective_idx_cpu.size()[-1]):
                    batch_idx = eos_effective_idx_cpu[i] // num_beams
                    if not done[batch_idx] :
                        generated_hyps[batch_idx.item()].add(
                            input_ids_cpu[eos_effective_idx_cpu[i]].clone(),
                            eos_effective_scores_cpu[i])
                    done[batch_idx] = (
                        done[batch_idx] or
                        generated_hyps[batch_idx].is_done(
                        next_scores[batch_idx].max().item(), cur_len))

            if all(done):
                break

            if use_reorder_cache_v2 and len(finished_batch_idxs) > 0:
                new_batch_size = batch_size - len(finished_batch_idxs)
                batch_mask = torch.ones(batch_size).to(next_tokens_id)
                batch_mask[torch.tensor(finished_batch_idxs)] = 0
                batch_idxs = batch_mask.nonzero().squeeze(-1)
                eos_mask = eos_mask[batch_idxs]
                next_beams_id = next_beams_id[batch_idxs]
                beams_offset.resize_(new_batch_size, 1)
                effective_beam_id = next_beams_id.add(beams_offset)
                next_scores = next_scores[batch_idxs]
                next_tokens = next_tokens[batch_idxs]
                next_tokens_id = next_tokens_id[batch_idxs]
                input_ids = input_ids.view(batch_size, -1)[batch_idxs].view(new_batch_size * num_beams, -1)
                before_batch_size = batch_size
                batch_size = new_batch_size
            else:
                before_batch_size = batch_size
                batch_idxs = None

            active_mask = torch.add(eos_mask.type_as(cand_offsets) * cand_size, cand_offsets[:eos_mask.size(1)])
            _, active_hypos = torch.topk(
                active_mask, k=num_beams, dim=1, largest=False)
            active_effective_beam_id  = torch.gather(
                effective_beam_id, dim=1, index=active_hypos)
            active_scores  = torch.gather(next_scores,
                dim=1, index=active_hypos)
            active_tokens  = torch.gather(next_tokens_id,
                dim=1, index=active_hypos)
            beam_idxs = active_effective_beam_id.view(-1)
            beam_scores = active_scores.view(-1)
            beam_tokens = active_tokens.view(-1)

            # re-order batch and update current length
            input_ids = input_ids[beam_idxs, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

            # re-order internal states
            if past is not None:
                if use_reorder_cache_v2:
                    new_beam_idxs = torch.arange(before_batch_size * num_beams).reshape(before_batch_size, num_beams).to(input_ids)
                    beam_idxs = new_beam_idxs[batch_idxs].reshape(-1)[beam_idxs]
                    past = self._reorder_cache_v2(past, batch_idxs, beam_idxs)
                else:
                    past = self._reorder_cache(past, beam_idxs)

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask,
                     attention_mask.new_ones((attention_mask.shape[0], 1))],
                    dim=-1)

        # finalize all open beam hypotheses and add to generated hypotheses
        unfin_offset = np.array(list(accumulate(done)))[np.array(done) == 0]
        if use_reorder_cache_v2:
            batch_size = len(unfin_offset)
        for batch_idx in range(batch_size):
            if not use_reorder_cache_v2 and done[batch_idx]:
                continue
            # test that beam scores match previously calculated scores if not
            # eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id
                 for token_id in next_tokens[batch_idx]):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] ==
                    beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: \
                {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx],
                    beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            if use_reorder_cache_v2:
                final_batch_idx = batch_idx + unfin_offset[batch_idx]
            else:
                final_batch_idx = batch_idx
            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[final_batch_idx].add(final_tokens, final_score)

        batch_size = len(done)

        # depending on whether greedy generation is wanted or not define
        # different output_batch_size and output_num_return_sequences_per_batch
        output_batch_size = batch_size if do_sample \
            else batch_size * num_return_sequences
        output_num_return_sequences_per_batch = 1 \
            if do_sample else num_return_sequences

        # select the best hypotheses
        sent_lengths = input_ids.new(output_batch_size)
        best = []

        # retrieve best hypotheses
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.beams, key=lambda x: x[0])
            for j in range(output_num_return_sequences_per_batch):
                effective_batch_idx = \
                    output_num_return_sequences_per_batch * i + j
                best_hyp = sorted_hyps.pop()[1]
                sent_lengths[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)

        # shorter batches are padded
        if sent_lengths.min().item() != sent_lengths.max().item():
            assert pad_token_id is not None, \
                "`Pad_token_id` has to be defined"
            sent_max_len = min(sent_lengths.max().item() + 1, max_length)
            decoded = input_ids.new(output_batch_size,
                sent_max_len).fill_(pad_token_id)

            # fill with hypothesis and eos_token_id if necessary
            for i, hypo in enumerate(best):
                decoded[i, : sent_lengths[i]] = hypo
                if sent_lengths[i] < max_length:
                    decoded[i, sent_lengths[i]] = eos_token_id
        else:
            # none of the hypotheses have an eos_token
            assert (len(hypo) == max_length for hypo in best)
            decoded = torch.stack(best).type(torch.long)\
                    .to(next(self.parameters()).device)

        return decoded
