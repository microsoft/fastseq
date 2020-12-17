# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Preprocess the raw text file by using bert tokenizer and generate the bpe files"""

import argparse

from nltk.tokenize.treebank import TreebankWordDetokenizer
import tqdm
from pytorch_transformers import BertTokenizer

def preocess(fin, fout, keep_sep=False, max_len=512):
    """Preprocess the raw text file by using bert tokenizer and generate the bpe
       files

    Args:
        fin (str): input raw text file path
        fout (str): output bpe file path
        keep_sep (bool, optional): indicates if the output strings will be
            joined with [X_SEP]. Defaults to False.
        max_len (int, optional): max input sentence length. Defaults to 512.
    """
    fin = open(fin, 'r', encoding='utf-8')
    fout = open(fout, 'w', encoding='utf-8')
    twd = TreebankWordDetokenizer()
    bpe = BertTokenizer.from_pretrained('bert-base-uncased')
    for line in tqdm.tqdm(fin.readlines()):
        line = line.strip().replace('``', '"').replace('\'\'', '"').replace(
            '`', '\'')
        s_list = [twd.detokenize(x.strip().split(
            ' '), convert_parentheses=True) for x in line.split('<S_SEP>')]
        tk_list = [bpe.tokenize(s) for s in s_list]
        output_string_list = [" ".join(s) for s in tk_list]
        if keep_sep:
            output_string = " [X_SEP] ".join(output_string_list)
        else:
            output_string = " ".join(output_string_list)
        output_string = " ".join(output_string.split(' ')[:max_len-1])
        fout.write('{}\n'.format(output_string))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin", type=str, required=True, help='input raw text file path')
    parser.add_argument("--fout", type=str, required=True, help='output bpe file path')
    parser.add_argument("--keep_sep", default=False, action="store_true",
        help='if True, the output strings will be joined with [X_SEP]')
    parser.add_argument("--max_len", type=int, default=1024,
        help="max input sentence length")

    args = parser.parse_args()
    preocess(args.fin, args.fout, args.keep_sep, args.max_len)

if __name__ == "__main__":
    main()
