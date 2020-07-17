import os
import shutil
import tempfile

import requests
import tarfile

from absl import logging
from filelock import FileLock
from tqdm.auto import tqdm
from zipfile import ZipFile, is_zipfile

get_named_temp_file = tempfile.NamedTemporaryFile

get_temp_file = tempfile.TemporaryFile

get_temp_dir = tempfile.TemporaryDirectory


def make_dirs(path, mode=0o777, exist_ok=False):
    try:
        os.makedirs(path, mode, exist_ok)
    except OSError as error:
        logging.error(error)


def wget(url, target_file_handling):
    response = requests.get(url, stream=True)
    if response.status_code == 416:  # Range not satisfiable
        return
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
    lock_file = os.path.join(output_dir, '.lock')
    make_dirs(output_dir, exist_ok=True)
    with FileLock(lock_file):
        if is_zipfile(input_compressed_file):
            with ZipFile(input_compressed_file, "r") as zip_file:
                extracted_folder_name = zip_file.namelist()[0]
                extracted_dir = os.path.join(output_dir, extracted_folder_name)
                shutil.rmtree(extracted_dir, ignore_errors=True)
                zip_file.extractall(output_dir)
                zip_file.close()
                return extracted_dir
        elif tarfile.is_tarfile(input_compressed_file):
            tar_file = tarfile.open(input_compressed_file)
            extracted_folder_name = tar_file.getnames()[0]
            extracted_dir = os.path.join(output_dir, extracted_folder_name)
            shutil.rmtree(extracted_dir, ignore_errors=True)
            tar_file.extractall(output_dir)
            tar_file.close()
            return extracted_dir
        else:
            raise ValueError(
                "The input file {} is not supported yet, please input .zip or .tar.gz file"
                .format(input_compressed_file))
