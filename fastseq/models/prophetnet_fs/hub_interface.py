# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Hub interface for ProphetNet"""

import copy
import logging
from fastseq.logging import get_logger
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import encoders

logger = get_logger(__name__, logging.INFO)

class ProphetNetHubInterface(nn.Module):
    """A simple PyTorch Hub interface to BART.

    Usage: https://github.com/pytorch/fairseq/tree/master/examples/BART
    """

    def __init__(self, args, task, model):
        super().__init__()
        self.args = args
        self.task = task
        self.model = model

        self.bpe = encoders.build_bpe(args)

        self.max_positions = min(utils.resolve_max_positions(
            self.task.max_positions(),
            self.model.max_positions(),
        ))

        # this is useful for determining the device
        self.register_buffer(
            '_float_tensor', torch.tensor([0], dtype=torch.float)) #pylint: disable=not-callable

    @property
    def device(self):
        return self._float_tensor.device

    def encode(self,
               sentence: str,
               *addl_sentences,
               no_separator=True) -> torch.LongTensor:
        """
        Encode a sentence by following BERT.
        For example::
            >>> encode('hello world').tolist()
            [7592, 2088, 102]
            >>> encode(' world').tolist()
            [2088, 102]
            >>> encode('world').tolist()
            [2088, 2]
        """
        sentence = sentence.replace('``', '"').replace('\'\'', '"').replace(
            '`', '\'').lower()
        tokens = self.bpe.encode(sentence)
        if len(tokens.split(' ')) > self.max_positions:
            tokens = ' '.join(tokens.split(' ')[:self.max_positions])
        tokens += ' [SEP]'
        tokens = self.task.source_dictionary.encode_line(
            tokens, append_eos=False)
        return tokens.long()

    def decode(self, tokens: torch.LongTensor):
        """ decode tokens

        Args:
            tokens (torch.LongTensor): input tokens.

        Returns:
            list(str): a list of strs.
        """
        assert tokens.dim() == 1
        tokens = tokens.cpu().numpy()
        if tokens[0] == self.task.source_dictionary.bos():
            tokens = tokens[1:]  # remove <s>
        eos_mask = (tokens == self.task.source_dictionary.eos())
        doc_mask = eos_mask[1:] & eos_mask[:-1]
        sentences = np.split(tokens, doc_mask.nonzero()[0] + 1)
        sentences = [self.bpe.decode(
            self.task.source_dictionary.string(s)) for s in sentences]
        if len(sentences) == 1:
            return sentences[0]
        return sentences

    def _build_sample(self, src_tokens: List[torch.LongTensor]):
        """Build input samples"""
        # assert torch.is_tensor(src_tokens)
        dataset = self.task.build_dataset_for_inference(
            src_tokens,
            [x.numel() for x in src_tokens],
        )
        sample = dataset.collater(dataset)
        sample = utils.apply_to_sample(
            lambda tensor: tensor.to(self.device),
            sample
        )
        return sample

    def sample(self,
               sentences: List[str],
               beam: int = 1,
               verbose: bool = False,
               **kwargs) -> str:
        input = [self.encode(sentence) for sentence in sentences]
        hypos = self.generate(input, beam, verbose, **kwargs)
        return [self.decode(x['tokens']) for x in hypos]

    def generate(self,
                 tokens: List[torch.LongTensor],
                 beam: int = 5,
                 verbose: bool = False,
                 **kwargs) -> torch.LongTensor:
        """Generate a batch of translations"""

        sample = self._build_sample(tokens)

        # build generator using current args as well as any kwargs
        gen_args = copy.copy(self.args)
        gen_args.beam = beam
        for k, v in kwargs.items():
            setattr(gen_args, k, v)
        generator = self.task.build_generator([self.model], gen_args)
        translations = self.task.inference_step(
            generator,
            [self.model],
            sample,
            prefix_tokens=None,
        )

        if verbose:
            src_str_with_unk = self.string(tokens)
            logger.info("S\t{}".format(src_str_with_unk))

        def getarg(name, default):
            return getattr(gen_args, name, getattr(self.args, name, default))

        # Process top predictions
        hypos = [x[0] for x in translations]
        hypos = [v for _, v in sorted(zip(sample['id'].tolist(), hypos))]
        return hypos

    def extract_features(self,
                         tokens: torch.LongTensor,
                         return_all_hiddens: bool = False) -> torch.Tensor:
        """Similar to *forward* but only return features"""

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        if tokens.size(-1) > min(self.model.max_positions()):
            raise ValueError('tokens exceeds maximum length: {} > {}'.format(
                tokens.size(-1), self.model.max_positions()
            ))
        tokens.to(device=self.device),
        prev_output_tokens = tokens.clone()

        prev_output_tokens[:, 0] = tokens.gather(
            1,
            (tokens.ne(self.task.source_dictionary.pad()).sum(
                dim=1)- 1).unsqueeze(-1),
        ).squeeze()

        prev_output_tokens[:, 1:] = tokens[:, :-1]
        features, extra = self.model(
            src_tokens=tokens,
            src_lengths=None,
            prev_output_tokens=prev_output_tokens,
            features_only=True,
            return_all_hiddens=return_all_hiddens,
        )
        if return_all_hiddens:
            # convert from T x B x C -> B x T x C
            inner_states = extra['inner_states']
            return [inner_state.transpose(0, 1) for inner_state in inner_states]
        else:
            return features  # just the last layer's features

    def register_classification_head(
        self,
        name: str,
        num_classes: int = None,
        embedding_size: int = None,
        **kwargs
    ):
        self.model.register_classification_head(
            name,
            num_classes=num_classes,
            embedding_size=embedding_size,
            **kwargs)

    def predict(self,
                head: str,
                tokens: torch.LongTensor,
                return_logits: bool = False):
        """Run the predictions and return the scores."""

        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
        features = self.extract_features(tokens.to(device=self.device))
        sentence_representation = features[
            tokens.eq(self.task.source_dictionary.eos()), :
        ].view(features.size(0), -1, features.size(-1))[:, -1, :]

        logits = self.model.classification_heads[head](sentence_representation)
        if return_logits:
            return logits
        return F.log_softmax(logits, dim=-1)
