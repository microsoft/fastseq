# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Test fastseq-generate-for-transformers related functions.
"""

from absl.testing import parameterized

from fastseq.utils.test_utils import fastseq_test_main, TestCaseBase
from fastseq_cli.transformers_generate import sort_sentences, unsort_sentences

class FastseqGenerateForTransformersTest(TestCaseBase):
    """Test fastseq-generate-for-transformers"""

    @parameterized.named_parameters(
        {'testcase_name': 'Ascending', 'reverse': False},
        {'testcase_name': 'Descending', 'reverse': True})
    def test_sort_unsort_sentences(self, reverse):
        """Test sort and unsort functions."""
        with open('tests/cli/data/val.source', mode='r') as file:
            text = file.readlines()

            sorted_text, sorted_idx = sort_sentences(text, reverse=reverse)
            restored_text = unsort_sentences(sorted_text, sorted_idx)

            self.assertNotEqual(text, sorted_text)
            self.assertEqual(text, restored_text)
            # check the memory address to make sure no extra copies.
            for i, _ in enumerate(text):
                self.assertEqual(id(text[i]), id(restored_text[i]))
            sorted_lens = [len(s) for s in sorted_text]
            expected_lens = sorted(sorted_lens, reverse=reverse)
            self.assertEqual(sorted_lens, expected_lens)

if __name__ == "__main__":
    fastseq_test_main()
