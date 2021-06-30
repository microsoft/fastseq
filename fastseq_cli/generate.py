# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Translate pre-processed data with a trained model and the optimizations provided
by FastSeq.
"""

import os
import sys

from fairseq import options

def parse_additional_args():
    parser = options.get_generation_parser()
    parser.add_argument(
        '--use-el-attn',
        action='store_true',
        help='Use Efficient Lossless Attention optimization ? ')
    args = options.parse_args_and_arch(parser)
    return args

def cli_main():
    os.environ['USE_EL_ATTN'] = '1' if '--use-el-attn' in sys.argv else '0'
    from fastseq.optimizer.fairseq.generate import main_v1  # pylint: disable=import-outside-toplevel
    args = parse_additional_args()
    main_v1(args)

if __name__ == '__main__':
    cli_main()
