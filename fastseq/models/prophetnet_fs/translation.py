# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Translation Task for ProphetNet."""

from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from fastseq.models.prophetnet_fs.bert_dictionary import BertDictionary

@register_task('translation_prophetnet')
class TranslationProphetnetTask(TranslationTask):
    """Translate from one (source) language to another (target) language."""

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)
