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
        '--use_el_attn',
        action='store_true',
        help='Use Efficient Lossless Attention optimization ? ')
    parser.add_argument(
        '--postprocess_workers',
        default=1,
        type=int,
        choices=range(1, 128, 1),
        metavar='N',
        help='number of worker for post process')
    parser.add_argument(
        '--decode_hypothesis',
        action="store_true")
    args = options.parse_args_and_arch(parser)
    return args

def cli_main():
    os.environ['USE_EL_ATTN'] = '1' if '--use-el-attn' in sys.argv else '0'
    from fastseq.optimizer.fairseq.generate import main_v2  # pylint: disable=import-outside-toplevel
    args = parse_additional_args()
    main_v2(args)

if __name__ == '__main__':
    cli_main()
