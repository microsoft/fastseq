# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Translate pre-processed data with a trained model and the optimizations provided
by FastSeq.
"""

from fairseq import options
import os
import argparse

def parse_additional_args ():
    parser = options.get_generation_parser()
    parser.add_argument('--use-el-attn', action='store_true',
            help='Use Efficient Lossless Attention optimization ? ')
    parser.add_argument(
        '--postprocess-workers',
        default=1,
        type=int,
        choices=range(1, 128, 1),
        metavar='N',
        help='number of worker for post process')
    parser.add_argument(
        '--decode-hypothesis',
        action="store_true")
    args = options.parse_args_and_arch(parser)
    return args

def cli_main():
    args = parse_additional_args()
    os.environ['USE_EL_ATTN'] = '1' if args.use_el_attn else '0'
    import fastseq
    from fastseq.optimizer.fairseq.generate import main_v1
    main_v1(args)
 
if __name__ == '__main__':
    cli_main()

