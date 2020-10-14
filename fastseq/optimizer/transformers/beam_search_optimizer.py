# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimization for beam search related parts in Transformers."""

import logging
from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F

from fastseq.utils.api_decorator import replace
from transformers.configuration_auto import BartConfig
from transformers.generation_utils import calc_banned_ngram_tokens, calc_banned_bad_words_ids, GenerationMixin, BeamHypotheses, top_k_top_p_filtering
from transformers.modeling_auto import MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
from transformers.modeling_bart import BartForConditionalGeneration, SelfAttention, _reorder_buffer
from transformers.modeling_t5 import T5ForConditionalGeneration
from fastseq.optimizer.ngram_repeat_block import NGramRepeatBlock

logger = logging.getLogger(__name__)

@replace(calc_banned_ngram_tokens)
def calc_banned_ngram_tokens_v2(prev_input_ids: Tensor,
                                num_hypos: int,
                                no_repeat_ngram_size: int,
                                cur_len: int,
                                pad_token_id: int) -> None:
    """Copied from fairseq for no_repeat_ngram in beam_search"""

    if cur_len + 1 < no_repeat_ngram_size:
        # return no banned tokens if we haven't generated no_repeat_ngram_size
        # tokens yet
        return [[] for _ in range(num_hypos)]
    generated_ngrams = [{} for _ in range(num_hypos)]
    for idx in range(num_hypos):
        gen_tokens = prev_input_ids[idx].tolist()
        generated_ngram = generated_ngrams[idx]
        for ngram in zip(
            *[gen_tokens[i:] for i in range(no_repeat_ngram_size)]):
            if ngram[-1] != pad_token_id:
                prev_ngram_tuple = tuple(ngram[:-1])
                generated_ngram[prev_ngram_tuple] = generated_ngram.get(
                    prev_ngram_tuple, []) + [ngram[-1]]

    def _get_generated_ngrams(hypo_idx):
        # Before decoding the next token, prevent decoding of ngrams that have
        # already appeared
        start_idx = cur_len + 1 - no_repeat_ngram_size
        ngram_idx = tuple(prev_input_ids[hypo_idx, start_idx:cur_len].tolist())
        return generated_ngrams[hypo_idx].get(ngram_idx, [])

    banned_tokens = [_get_generated_ngrams(hypo_idx)
                     for hypo_idx in range(num_hypos)]
    return banned_tokens



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
            "only {max_length}. Please make sure that `max_length` is bigger "
            "than the number of tokens, by setting either "
            "`generate(max_length=...,...)` or `config.max_length = ...`")

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
            scores = self.no_repeat_ngram_op(input_ids,scores.float(),
                    batch_size, cur_len, num_beams, no_repeat_ngram_size)

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

        # for greedy decoding it is made sure that only tokens of the first beam are considered to avoid sampling the exact same tokens three times
        if do_sample is False:
            beam_scores[:, 1:] = -1e9
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # cache compute states
        past = (encoder_outputs, None) if encoder_outputs is not None else None

        # done sentences
        done = [False for _ in range(batch_size)]

        #NGram Repeat block Op
        self.no_repeat_ngram_op = NGramRepeatBlock()#.to('cuda', torch.float32)

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(
                input_ids, past=past, attention_mask=attention_mask,
                use_cache=use_cache, **model_specific_kwargs
            )
            outputs = self(**model_inputs)  # (batch_size * num_beams, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size * num_beams, vocab_size)

            # if model has past, then set the past variable to speed up decoding
            if self._use_cache(outputs, use_cache):
                past = outputs[1]
            if self.config.is_encoder_decoder and do_sample is False:
                # TODO (PVP) still a bit hacky here - there might be a better solution
                next_token_logits = self.adjust_logits_during_generation(
                    next_token_logits, cur_len=cur_len, max_length=max_length
                )

            scores = F.log_softmax(next_token_logits, dim=-1)  # (batch_size * num_beams, vocab_size)
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
                curr_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)
                # Temperature
                if temperature != 1.0:
                    curr_scores = curr_scores / temperature
                # Top-p/top-k filtering
                curr_scores = top_k_top_p_filtering(
                    curr_scores, top_k=top_k, top_p=top_p, min_tokens_to_keep=2
                )  # (batch_size * num_beams, vocab_size)
                # re-organize to group the beam together to sample from all beam_idxs
                curr_scores = curr_scores.contiguous().view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                # Sample 2 next tokens for each beam (so we have some spare tokens and match output of greedy beam search)
                probs = F.softmax(curr_scores, dim=-1)
                next_tokens = torch.multinomial(probs,
                            num_samples=2 * num_beams)  # (batch_size, num_beams * 2)
                # Compute next scores
                next_scores = torch.gather(curr_scores, -1, next_tokens)  # (batch_size, num_beams * 2)
                # sort the sampled vector to make sure that the first num_beams samples are the best
                next_scores, next_scores_indices = torch.sort(
                                    next_scores, descending=True, dim=1)
                next_tokens = torch.gather(next_tokens, -1, next_scores_indices)  # (batch_size, num_beams * 2)

            else:
                next_scores = scores + beam_scores[:, None].expand_as(scores)  # (batch_size * num_beams, vocab_size)

                # re-organize to group the beam together (we are keeping top hypothesis accross beams)
                next_scores = next_scores.view(
                    batch_size, num_beams * vocab_size
                )  # (batch_size, num_beams * vocab_size)

                next_scores, next_tokens = torch.topk(next_scores,
                                2 * num_beams, dim=1, largest=True, sorted=True)

            assert next_scores.size() == next_tokens.size() \
                == (batch_size, 2 * num_beams)
            # next batch beam content
            next_batch_beam = []
            next_tokens_id = next_tokens % vocab_size
            next_beams_id = next_tokens // vocab_size
            beams_offset = (torch.arange(0, batch_size) * num_beams)\
                .unsqueeze(1).type_as(next_beams_id)
            effective_beam_id =  next_beams_id + beams_offset
            if eos_token_id is not None :
                eos_mask = next_tokens_id.eq(eos_token_id)
            else :
                eos_mask = torch.zeros_like(next_tokens_id).bool()
            eos_effective_idx = torch.masked_select(
                effective_beam_id[:, :num_beams], mask=eos_mask[:, :num_beams]
            )
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
                                eos_effective_scores_cpu[i],
                            )
                done[batch_idx] = (done[batch_idx] or
                    generated_hyps[batch_idx].is_done(
                    next_scores[batch_idx].max().item(), cur_len
                        ))
            cand_offsets = (torch.arange(0, 2 * num_beams)
                            .type_as(input_ids).cuda())
            active_mask = torch.add(
                eos_mask.type_as(cand_offsets) * (2*num_beams),
                cand_offsets[: eos_mask.size(1)],
            )
            _, active_hypos = torch.topk(
                active_mask, k=num_beams, dim=1, largest=False
            )
            active_effective_beam_id  = torch.gather(
                effective_beam_id, dim=1, index=active_hypos)
            active_scores  = torch.gather(next_scores,
                dim=1, index=active_hypos)
            active_tokens  = torch.gather(next_tokens_id,
                dim=1, index=active_hypos)
            beam_idx = active_effective_beam_id.view(-1)
            beam_scores = active_scores.view(-1)
            beam_tokens = active_tokens.view(-1)

            #stop when we are done with each sentence
            if all(done):
                break
            # re-order batch and update current length
            input_ids = input_ids[beam_idx, :]
            input_ids = torch.cat([input_ids, beam_tokens.unsqueeze(1)], dim=-1)
            cur_len = cur_len + 1

            # re-order internal states
            if past is not None:
                past = self._reorder_cache(past, beam_idx)

            # extend attention_mask for new generated input if only decoder
            if self.config.is_encoder_decoder is False:
                attention_mask = torch.cat(
                    [attention_mask,
                attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
                )

        # finalize all open beam hypotheses and add to generated hypotheses
        for batch_idx in range(batch_size):
            if done[batch_idx]:
                continue

            # test that beam scores match previously calculated scores if not eos and batch_idx not done
            if eos_token_id is not None and all(
                (token_id % vocab_size).item() != eos_token_id
                for token_id in next_tokens[batch_idx]
            ):
                assert torch.all(
                    next_scores[batch_idx, :num_beams] ==
                    beam_scores.view(batch_size, num_beams)[batch_idx]
                ), "If batch_idx is not done, final next scores: \
                {} have to equal to accumulated beam_scores: {}".format(
                    next_scores[:, :num_beams][batch_idx],
                    beam_scores.view(batch_size, num_beams)[batch_idx],
                )

            # need to add best num_beams hypotheses to generated hyps
            for beam_id in range(num_beams):
                effective_beam_id = batch_idx * num_beams + beam_id
                final_score = beam_scores[effective_beam_id].item()
                final_tokens = input_ids[effective_beam_id]
                generated_hyps[batch_idx].add(final_tokens, final_score)

        # depending on whether greedy generation is wanted or not define different output_batch_size and output_num_return_sequences_per_batch
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
        ((enc_out, enc_mask), decoder_past_key_values) = past
        reordered_past = []
        for layer_past in decoder_past_key_values:
            # Get the correct batch idx from decoder layer's batch dim for
            # self-attn; Note that there is no need to reorder the cached key
            # and value for the encoder-decoder-attn, because the key and value
            # for the beams of each sample is the same and we can cache just one
            # copy to save GPU memory.
            layer_past_new = {}
            for attn_key, attn_cache in layer_past.items():
                if attn_key == 'self':
                    layer_past_new[attn_key] = _reorder_buffer(
                        attn_cache, beam_idx)
                    continue
                layer_past_new[attn_key] = attn_cache

            reordered_past.append(layer_past_new)

        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(
            0, beam_idx)

        past = ((enc_out, new_enc_mask), reordered_past)
        return past

MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING[BartConfig] = BartForConditionalGenerationV2 # pylint: disable=line-too-long
