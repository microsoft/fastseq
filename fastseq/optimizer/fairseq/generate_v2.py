import logging
import math
import os
import sys

import numpy as np

import torch

from fairseq import checkpoint_utils, options, scoring, tasks, utils
from fairseq_cli.generate import _main, get_symbols_to_strip_from_output
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fairseq.data import encoders

from fastseq.utils.api_decorator import register_fairseq_optimized_class, replace


@replace(_main)
def _main_v2(args, output_file):
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=output_file,
    )
    logger = logging.getLogger('fastseq_cli.generate')

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    logger.info(args)

    # Fix seed for stochastic decoding
    if args.seed is not None and not args.no_seed_provided:
        np.random.seed(args.seed)
        utils.set_torch_seed(args.seed)

    use_cuda = torch.cuda.is_available() and not args.cpu

    # Load dataset splits
    task = tasks.setup_task(args)
    task.load_dataset(args.gen_subset)

    # Set dictionaries
    try:
        src_dict = getattr(task, 'source_dictionary', None)
    except NotImplementedError:
        src_dict = None
    tgt_dict = task.target_dictionary

    # Load ensemble
    logger.info('loading model(s) from {}'.format(args.path))
    models, _model_args = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides=eval(args.model_overrides),
        task=task,
        suffix=getattr(args, "checkpoint_suffix", ""),
    )

    # Optimize ensemble for generation
    for model in models:
        model.prepare_for_inference_(args)
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

    # Load alignment dictionary for unknown word replacement
    # (None if no unknown word replacement, empty if no path to align dictionary)
    align_dict = utils.load_align_dict(args.replace_unk)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_tokens=args.max_tokens,
        max_sentences=args.max_sentences,
        max_positions=utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]),
        ignore_invalid_inputs=args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple=args.required_batch_size_multiple,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
        num_workers=args.num_workers,
    ).next_epoch_itr(shuffle=False)
    progress = progress_bar.progress_bar(
        itr,
        log_format=args.log_format,
        log_interval=args.log_interval,
        default_log_format=('tqdm' if not args.no_progress_bar else 'none'),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(models, args)

    # Handle tokenization and BPE
    tokenizer = encoders.build_tokenizer(args)
    bpe = encoders.build_bpe(args)

    def decode_fn(x):
        if bpe is not None:
            x = bpe.decode(x)
        if tokenizer is not None:
            x = tokenizer.decode(x)
        return x

    scorer = scoring.scoring_utils.build_scorer(args, tgt_dict)

    num_sentences = 0
    has_target = True
    wps_meter = TimeMeter()
    for sample in progress:
        cpu_sample = sample
        if 'net_input' not in sample:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = sample['target'][:, :args.prefix_size]

        gen_timer.start()
        hypos = task.inference_step(generator, models, sample, prefix_tokens)
        hypos = utils.move_to_cpu(hypos) if use_cuda else hypos
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
        gen_timer.stop(num_generated_tokens)

        sample = cpu_sample
        for i, sample_id in enumerate(sample['id'].tolist()):
            has_target = sample['target'] is not None

            # Remove padding
            if 'src_tokens' in sample['net_input']:
                src_tokens = utils.strip_pad(
                    sample['net_input']['src_tokens'][i, :], tgt_dict.pad())
            else:
                src_tokens = None

            target_tokens = None
            if has_target:
                target_tokens = utils.strip_pad(sample['target'][i, :],
                                                tgt_dict.pad()).int()

            # Either retrieve the original sentences or regenerate them from tokens.
            if align_dict is not None:
                src_str = task.dataset(
                    args.gen_subset).src.get_original_text(sample_id)
                target_str = task.dataset(
                    args.gen_subset).tgt.get_original_text(sample_id)
            else:
                if src_dict is not None:
                    src_str = src_dict.string(src_tokens, args.remove_bpe)
                else:
                    src_str = ""
                if has_target:
                    target_str = tgt_dict.string(
                        target_tokens,
                        args.remove_bpe,
                        escape_unk=True,
                        extra_symbols_to_ignore=
                        get_symbols_to_strip_from_output(generator),
                    )

            src_str = decode_fn(src_str)
            if has_target:
                target_str = decode_fn(target_str)

            if not args.quiet:
                if src_dict is not None:
                    print(
                        'S-{}\t{}'.format(sample_id, src_str),
                        file=output_file)
                if has_target:
                    print(
                        'T-{}\t{}'.format(sample_id, target_str),
                        file=output_file)

            # Process top predictions
            for j, hypo in enumerate(hypos[i][:args.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens=hypo['tokens'].int(),
                    src_str=src_str,
                    alignment=hypo['alignment'],
                    align_dict=align_dict,
                    tgt_dict=tgt_dict,
                    remove_bpe=args.remove_bpe,
                    extra_symbols_to_ignore=get_symbols_to_strip_from_output(
                        generator),
                )
                detok_hypo_str = decode_fn(hypo_str)
                if not args.quiet:
                    score = hypo['score'] / math.log(2)  # convert to base 2
                    # original hypothesis (after tokenization and BPE)
                    print(
                        'H-{}\t{}\t{}'.format(sample_id, score, hypo_str),
                        file=output_file)
                    # detokenized hypothesis
                    print(
                        'D-{}\t{}\t{}'.format(sample_id, score,
                                              detok_hypo_str),
                        file=output_file)
                    print(
                        'P-{}\t{}'.format(
                            sample_id,
                            ' '.join(
                                map(
                                    lambda x: '{:.4f}'.format(x),
                                    # convert from base e to base 2
                                    hypo['positional_scores'].div_(
                                        math.log(2)).tolist(),
                                ))),
                        file=output_file)

                    if args.print_alignment:
                        print(
                            'A-{}\t{}'.format(sample_id, ' '.join([
                                '{}-{}'.format(src_idx, tgt_idx)
                                for src_idx, tgt_idx in alignment
                            ])),
                            file=output_file)

                    if args.print_step:
                        print(
                            'I-{}\t{}'.format(sample_id, hypo['steps']),
                            file=output_file)

                    if getattr(args, 'retain_iter_history', False):
                        for step, h in enumerate(hypo['history']):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens=h['tokens'].int(),
                                src_str=src_str,
                                alignment=None,
                                align_dict=None,
                                tgt_dict=tgt_dict,
                                remove_bpe=None,
                            )
                            print(
                                'E-{}_{}\t{}'.format(sample_id, step, h_str),
                                file=output_file)

                # Score only the top hypothesis
                if has_target and j == 0:
                    if align_dict is not None or args.remove_bpe is not None:
                        # Convert back to tokens for evaluation with unk replacement and/or without BPE
                        target_tokens = tgt_dict.encode_line(
                            target_str, add_if_not_exist=True)
                        hypo_tokens = tgt_dict.encode_line(
                            detok_hypo_str, add_if_not_exist=True)
                    if hasattr(scorer, 'add_string'):
                        scorer.add_string(target_str, detok_hypo_str)
                    else:
                        scorer.add(target_tokens, hypo_tokens)

        wps_meter.update(num_generated_tokens)
        progress.log({'wps': round(wps_meter.avg)})
        num_sentences += sample[
            "nsentences"] if "nsentences" in sample else sample['id'].numel()

    logger.info('NOTE: hypothesis and token scores are output in base 2')
    logger.info(
        'Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'.
        format(num_sentences, gen_timer.n, gen_timer.sum,
               num_sentences / gen_timer.sum, 1. / gen_timer.avg))
    if has_target:
        if args.bpe and not args.sacrebleu:
            if args.remove_bpe:
                logger.warning(
                    "BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization"
                )
            else:
                logger.warning(
                    "If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization"
                )
        # use print to be consistent with other main outputs: S-, H-, T-, D- and so on
        print(
            'Generate {} with beam={}: {}'.format(args.gen_subset, args.beam,
                                                  scorer.result_string()),
            file=output_file)

    return scorer
