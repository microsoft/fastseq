# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Test the file_util functions."""

import os

from absl.testing import absltest, parameterized

from fastseq.utils.file_utils import decompress_file, get_temp_dir, make_dirs, wget
from fastseq.utils.test_utils import fastseq_test_main, TestCaseBase


class FileUtilsTest(TestCaseBase):
    """Test the file_util functions."""

    def setUp(self):
        self.parent_dir_handle = get_temp_dir()
        self.parent_dir = self.parent_dir_handle.name

    def tearDown(self):
        self.parent_dir_handle.cleanup()

    @parameterized.named_parameters({
        'testcase_name': 'Normal',
        'directory': 'test_dir',
        'mode': 0o777,
        'exist_ok': False,
    })
    def test_make_dirs(self, directory, mode, exist_ok):
        """Test `make_dirs()`

        Args:
            directory (str): file folder.
            mode (int): directory mode.
            exist_ok (bool): indicate whether it is ok if the input directory
                             exists.
        """
        path = os.path.join(self.parent_dir, directory)
        make_dirs(path, mode, exist_ok)
        self.assertTrue(os.path.exists(path), True)

    @parameterized.named_parameters({
        'testcase_name': 'Normal',
        'url':
        'https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz',
        'target_file_name': 'bart.base.tar.gz',
    })
    def disable_test_wget(self, url, target_file_name):
        """Test `wget()`.

        It is disabled because it is time consuming to download the model file.
        Once we find a small test file, it will be enabled. Currently, `wget()`
        tests are coverred by `test_wget_and_decompress_file()`.

        Args:
            url (str): download url.
            target_file_name (target): the expected file name.
        """
        target_file = os.path.join(self.parent_dir, target_file_name)
        with open(target_file, "xb") as output_file:
            wget(url, output_file)
        self.assertTrue(os.path.exists(target_file), True)

    @parameterized.named_parameters({
        'testcase_name': 'Normal',
        'tar_file_url':
        'https://dl.fbaipublicfiles.com/fairseq/models/bart.base.tar.gz',
        'tar_file_name': 'bart.base.tar.gz',
        'output_folder': 'tar_dir',
    })
    def test_wget_and_decompress_file(self, tar_file_url, tar_file_name,
                                      output_folder):
        """Test `wget()` and `decompress_file().

        Args:
            tar_file_url (str): download url for tar file.
            tar_file_name (str): tar file name.
            output_folder (str): directory for decompressing the tar file.
        """
        # download the tar file.
        tar_file_path = os.path.join(self.parent_dir, tar_file_name)
        with open(tar_file_path, "xb") as tar_file:
            wget(tar_file_url, tar_file)

        # decompress the tar file.
        output_dir = os.path.join(self.parent_dir, output_folder)
        output_file = decompress_file(tar_file_path, output_dir)
        self.assertTrue(os.path.exists(output_file), True)


if __name__ == "__main__":
    fastseq_test_main()
