# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility funcs to create, download, decompress files/packages/directories."""

import os
import shutil
import tarfile
import tempfile
from zipfile import ZipFile, is_zipfile

import requests
from filelock import FileLock
from tqdm.auto import tqdm

from fastseq.logging import get_logger

logger = get_logger(__name__)

get_named_temp_file = tempfile.NamedTemporaryFile

get_temp_file = tempfile.TemporaryFile

get_temp_dir = tempfile.TemporaryDirectory  # pylint: disable=invalid-name


def make_dirs(path, mode=0o777, exist_ok=False):
    """Create the specified directory.

    Args:
        path (str): the directory path
        mode (int, optional): the directory mode. Defaults to 0o777.
        exist_ok (bool, optional): whether it is ok if the input path exits.
                                   Defaults to False.
    """
    try:
        os.makedirs(path, mode, exist_ok)
    except OSError as error:
        logger.error(error)


def wget(url, target_file_handling):
    """Download the file from the url to the target file.

    Args:
        url (str): the url to download the data.
        target_file_handling (file object): A file handling for writing the
                                            data.
    """
    response = requests.get(url, stream=True)
    if response.status_code == 416:  # Range not satisfiable
        raise ValueError("Range not satisfiable when loading {}".format(url))
    content_length = response.headers.get("Content-Length")
    total = int(content_length) if content_length is not None else None
    progress = tqdm(unit="B",
                    unit_scale=True,
                    total=total,
                    initial=0,
                    desc="Downloading")
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:  # filter out keep-alive new chunks
            progress.update(len(chunk))
            target_file_handling.write(chunk)
    progress.close()


def decompress_file(input_compressed_file, output_dir):
    """Extract the compressed files in .tar.gz or .zip formats.

    Args:
        input_compressed_file (str): file path for the compressed package.
        output_dir (str): output directory for the decompressed files.

    Raises:
        ValueError: if the input file is not in .tar.gz or .zip formats.

    Returns:
        str: the direcotry path for the decompressed files, which will be the
             combination of `output_dir` and the name of input compressed
             package.
    """
    lock_file = os.path.join(output_dir, '.lock')
    make_dirs(output_dir, exist_ok=True)
    with FileLock(lock_file):
        if tarfile.is_tarfile(input_compressed_file):
            tar_file = tarfile.open(input_compressed_file)
            extracted_folder_name = tar_file.getnames()[0]
            extracted_dir = os.path.join(output_dir, extracted_folder_name)
            shutil.rmtree(extracted_dir, ignore_errors=True)
            tar_file.extractall(output_dir)
            tar_file.close()
            return extracted_dir

        # TODO: add a test case for .zip file.
        if is_zipfile(input_compressed_file):
            with ZipFile(input_compressed_file, "r") as zip_file:
                extracted_folder_name = zip_file.namelist()[0]
                extracted_dir = os.path.join(output_dir, extracted_folder_name)
                shutil.rmtree(extracted_dir, ignore_errors=True)
                zip_file.extractall(output_dir)
                zip_file.close()
                return extracted_dir

        raise ValueError(
            "The input file {} is not supported yet, please input .zip or .tar.gz file"  # pylint: disable=line-too-long
            .format(input_compressed_file))
