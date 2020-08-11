# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Translate pre-processed data with a trained model and the optimizations provided
by FastSeq.
"""

import fastseq

from fairseq_cli.generate import cli_main

if __name__ == '__main__':
    cli_main()
