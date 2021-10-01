# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimize fairseq-generate (v0.10.2)"""

import ast
import logging
import math
import os
import sys
from itertools import chain
from multiprocessing import Queue, JoinableQueue
from torch.multiprocessing import Process
import numpy as np
import torch
from fairseq_cli.generate import main
from fairseq.utils import apply_to_sample
from fairseq import scoring, checkpoint_utils, tasks, utils
from fairseq.logging import progress_bar
from fairseq.logging.meters import StopwatchMeter, TimeMeter
from fastseq.utils.api_decorator import replace
from fairseq.options import add_generation_args

GENERATE_FINISHED = "done"
POSTPROCESS_FINISHED = None

original_add_generation_args = add_generation_args

@replace(add_generation_args)
def add_generation_args_v1(parser):
    group = original_add_generation_args(parser)
    # fmt: off
    group.add_argument(
        '--postprocess-workers',
        default=1,
        type=int,
        choices=range(1, 128, 1),
        metavar='N',
        help='number of worker for post process')
    group.add_argument(
        '--decode-hypothesis',
        action="store_true", 
        default=True)
    group.add_argument(
        '--use-el-attn',
        action='store_true',
        help='Use Efficient Lossless Attention optimization ? ')
    # fmt: on

def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32.
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            return tensor.cpu().to(dtype=torch.float32)
        else:
            return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)

def convert_base_e_to_base_2(tensor):
    return tensor.div_(math.log(2))

class IOProcess(Process):
    """
    Single process to handle IO and compute metrics
    """

    def __init__(self, args, task, message_queue, output_file):
        """
        Process to handle IO and compute metrics

        Args:
            args (Namespace): paramerter for model and generation
            task (fairseq.tasks.fairseq_task.Fairseq):
                use to load dict for detokenize
            message_queue (multiprocessing.Queue): queue store output
        """
        super(IOProcess, self).__init__()
        self.tgt_dict = task.target_dictionary
        
        # Generate and compute BLEU score
        self.scorer = scoring.build_scorer(args, self.tgt_dict)
        self.args = args
        self.message_queue = message_queue
        self.has_target = False
        self.output_file = output_file
    
    def run(self):
        while True:
            msg = self.message_queue.get()
            if isinstance(msg, tuple):
                t, h = msg
                if hasattr(self.scorer, 'add_string'):
                    self.scorer.add_string(t, h)
                else:
                    self.scorer.add(t, h)
                self.has_target = True
            elif msg == GENERATE_FINISHED:
                if self.has_target:
                    if self.args.bpe and not self.args.sacrebleu:
                        if self.args.remove_bpe:
                            print("BLEU score is being computed by splitting detokenized string on spaces, this is probably not what you want. Use --sacrebleu for standard 13a BLEU tokenization")
                        else:
                            print("If you are using BPE on the target side, the BLEU score is computed on BPE tokens, not on proper words.  Use --sacrebleu for standard 13a BLEU tokenization")
                    print("Generate {} with beam={}: {}".format(
                        self.args.gen_subset, self.args.beam, self.scorer.result_string()),
                        file=self.output_file,)
                break
            else:
                print(msg, file = self.output_file)
            self.message_queue.task_done()
        self.message_queue.close()
        self.message_queue.join_thread()

class PostProcess(Process):
    """
    Use multiple processes to do detokenization
    """

    def __init__(self, args, task, data_queue, message_queue, generator):
        """
        Handle detokenize and belu score computation

        Args:
            args (Namespace): paramerter for model and generation
            task (fairseq.tasks.fairseq_task.Fairseq):
                use to load dict for detokenize
            data_queue (multiprocessing.Queue):
                queue store tensor data for detokenize
            message_queue (multiprocessing.Queue): queue store output
        """
        super(PostProcess, self).__init__()
        # Set dictionaries
        try:
            self.src_dict = getattr(task, 'source_dictionary', None)
        except NotImplementedError:
            self.src_dict = None
        self.tgt_dict = task.target_dictionary

        # Load alignment dictionary for unknown word replacement
        # (None if no unknown word replacement, empty if no path to align dictionary)
        self.align_dict = utils.load_align_dict(args.replace_unk)

        # Generate and compute BLEU score
        self.scorer = scoring.build_scorer(args, self.tgt_dict)
        self.args = args
        self.task = task
        self.data_queue = data_queue
        self.message_queue = message_queue
        self.generator = generator
        if args.decode_hypothesis:
            self.tokenizer = task.build_tokenizer(args)
            self.bpe = task.build_bpe(args)

    def _decode(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x

    def _get_symbols_to_strip_from_output(self, generator):
        if hasattr(generator, "symbols_to_strip_from_output"):
            return generator.symbols_to_strip_from_output
        else:
            return {generator.eos}

    def _detokenize(self, sample, hypos):
        """ 
        Detokenize and compute BELU
        """
        message_list = []
        for i, sample_id in enumerate(sample['id'].tolist()):
            has_target = sample['target'] is not None

            # Remove padding
            if "src_tokens" in sample["net_input"]:
                src_tokens = utils.strip_pad(
                    sample["net_input"]["src_tokens"][i, :], self.tgt_dict.pad()
                )
            else:
                src_tokens = None
            target_tokens = None
            if has_target:
                target_tokens = (
                    utils.strip_pad(sample["target"][i, :], self.tgt_dict.pad()).int().cpu()
                )

            # Either retrieve the original sentences or regenerate them from tokens
            if self.align_dict is not None:
                src_str = self.task.dataset(
                    self.args.gen_subset).src.get_original_text(sample_id)
                target_str = self.task.dataset(
                    self.args.gen_subset).tgt.get_original_text(sample_id)
            else:
                if self.src_dict is not None:
                    src_str = self.src_dict.string(src_tokens,
                                                   self.args.remove_bpe)
                else:
                    src_str = ""
                if has_target:
                    target_str = self.tgt_dict.string(
                        target_tokens, 
                        self.args.remove_bpe, 
                        escape_unk = True,
                        extra_symbols_to_ignore = self._get_symbols_to_strip_from_output(self.generator),
                        )
            if not self.args.quiet:
                if self.src_dict is not None:
                    if self.args.decode_hypothesis:
                        message_list.append('S-{}\t{}'.format(
                            sample_id, self._decode(src_str)))
                    else:
                        message_list.append('S-{}\t{}'.format(
                            sample_id, src_str))
                if has_target:
                    if self.args.decode_hypothesis:
                        message_list.append('T-{}\t{}'.format(
                            sample_id, self._decode(target_str)))
                    else:
                        message_list.append('T-{}\t{}'.format(
                            sample_id, target_str))
        
            # Process top predictions
            for j, hypo in enumerate(hypos[i][:self.args.nbest]):
                hypo_tokens, hypo_str, alignment = utils.post_process_prediction(
                    hypo_tokens = hypo['tokens'].int(),
                        src_str = src_str,
                        alignment = hypo['alignment'],
                        align_dict = self.align_dict,
                        tgt_dict = self.tgt_dict,
                        remove_bpe = self.args.remove_bpe,
                        extra_symbols_to_ignore = self._get_symbols_to_strip_from_output(self.generator),
                    )
                if not self.args.quiet:
                    score = convert_base_e_to_base_2(hypo["score"])
                    message_list.append('H-{}\t{}\t{}'.format(
                        sample_id, score, hypo_str))
                    if self.args.decode_hypothesis:
                        detok_hypo_str = self._decode(hypo_str)
                        message_list.append('D-{}\t{}\t{}'.format(
                            sample_id, score, detok_hypo_str))
                    message_list.append('P-{}\t{}'.format(
                        sample_id, ' '.join(
                            map(
                                lambda x: '{:.4f}'.format(x),
                                convert_base_e_to_base_2(hypo['positional_scores']).tolist(),
                            ))))
                    if self.args.print_alignment:
                        message_list.append('A-{}\t{}'.format(
                            sample_id, ' '.join([
                                '{}-{}'.format(src_idx, tgt_idx)
                                for src_idx, tgt_idx in alignment
                            ])))
                    if self.args.print_step:
                        message_list.append('I-{}\t{}'.format(
                            sample_id, hypo['steps']))
                    if getattr(self.args, 'retain_iter_history', False):
                        for step, h in enumerate(hypo['history']):
                            _, h_str, _ = utils.post_process_prediction(
                                hypo_tokens = h['tokens'].int(),
                                src_str = self.src_str, 
                                alignment = None, 
                                align_dict = None,
                                tgt_dict = self.tgt_dict, 
                                remove_bpe = None,
                                )
                            message_list.append('E-{}_{}\t{}'.format(sample_id, step, h_str))
                
                # Score only the top hypothesis
                if has_target and j == 0:
                    if (self.align_dict is not None or
                        self.args.remove_bpe is not None):
                        
                        # Convert back to tokens for evaluation with unk
                        # replacement and/or without BPE
                        target_tokens = self.tgt_dict.encode_line(
                            target_str, add_if_not_exist = True)
                        hypo_tokens = self.tgt_dict.enode_line(
                            detok_hypo_str, add_if_not_exist = True)
                    if hasattr(self.scorer, "add_string"):
                        self.message_queue.put((target_str, detok_hypo_str))
                    else:
                        self.message_queue.put((target_tokens, hypo_tokens)) 
        self.message_queue.put('\n'.join(message_list))

    def run(self):
        while True:
            r = self.data_queue.get()
            if r == GENERATE_FINISHED or r is POSTPROCESS_FINISHED:
                self.data_queue.put(POSTPROCESS_FINISHED)
                break
            else:
                sample, hypos = r
                self._detokenize(sample, hypos)
        self.data_queue.close()
        self.data_queue.join_thread()
        self.message_queue.close()
        self.message_queue.join_thread()
        self.message_queue.join()

def _main(args, output_file):
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO, 
        stream=output_file,
    )
    logger = logging.getLogger("fastseq.optimizer.fairseq.generate")
    utils.import_user_module(args)
    if args.max_tokens is None and args.batch_size is None:
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
    overrides = ast.literal_eval(args.model_overrides)

    # Load ensemble
    logger.info("loading model(s) from {}".format(args.path))
    models, _ = checkpoint_utils.load_model_ensemble(
        utils.split_paths(args.path),
        arg_overrides = overrides,
        task = task,
        suffix = getattr(args, "checkpoint_suffix", ""),
        strict = (args.checkpoint_shard_count == 1),
        num_shards = args.checkpoint_shard_count,
    )
    if args.lm_path is not None:
        overrides["data"] = args.data
        try:
            lms, _ = checkpoint_utils.load_model_ensemble(
                [args.lm_path],
                arg_overrides=overrides,
                task=None,
            )
        except:
            logger.warning("Failed to load language model! Please make sure that the language model dict is the same as target dict and is located in the data dir ({})".format(args.data))
            raise
        assert len(lms) == 1
    else:
        lms = [None]

    # Optimize ensemble for generation
    for model in chain(models, lms):
        if model is None:
            continue
        if args.fp16:
            model.half()
        if use_cuda and not args.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(args)

    # Load dataset (possibly sharded)
    itr = task.get_batch_iterator(
        dataset = task.dataset(args.gen_subset),
        max_tokens = args.max_tokens,
        max_sentences = args.batch_size,
        max_positions = utils.resolve_max_positions(
            task.max_positions(),
            *[model.max_positions() for model in models]),
        ignore_invalid_inputs = args.skip_invalid_size_inputs_valid_test,
        required_batch_size_multiple = args.required_batch_size_multiple,
        num_shards = args.num_shards,
        shard_id = args.shard_id,
        num_workers = args.num_workers,
        data_buffer_size = args.data_buffer_size,
    ).next_epoch_itr(shuffle = False)
    progress = progress_bar.progress_bar(
        itr,
        log_format = args.log_format,
        log_interval = args.log_interval,
        default_log_format = ("tqdm" if not args.no_progress_bar else "none"),
    )

    # Initialize generator
    gen_timer = StopwatchMeter()
    extra_gen_cls_kwargs = {"lm_model": lms[0], "lm_weight": args.lm_weight}
    generator = task.build_generator(
        models, args, extra_gen_cls_kwargs = extra_gen_cls_kwargs
    )
    num_sentences = 0
    data_queue = Queue()
    message_queue = JoinableQueue()
    p_list = []
    for _ in range(args.postprocess_workers):
        p = PostProcess(args, task, data_queue, message_queue, generator)
        p_list.append(p)
        p.start()
    io_process = IOProcess(args, task, message_queue, output_file)
    io_process.start()
    if args.use_el_attn:
        task.transpose_enc_dec_kv_proj(models)
    
    wps_meter = TimeMeter()
    for sample in progress:
        cpu_sample = sample
        if 'net_input' not in sample:
            continue
        sample = utils.move_to_cuda(sample) if use_cuda else sample

        prefix_tokens = None
        if args.prefix_size > 0:
            prefix_tokens = sample['target'][:, :args.prefix_size]

        constraints = None
        if "constraints" in sample:
            constraints = sample["constraints"]

        gen_timer.start()
        try:
            hypos = task.inference_step(
                generator, models, sample, prefix_tokens, constraints)
        except:
            logging.exception(sys.exc_info()[0])
            for p in p_list:
                p.terminate()
            io_process.terminate()
            data_queue.close()
            message_queue.close()
            sys.exit(1)
        num_generated_tokens = sum(len(h[0]['tokens']) for h in hypos)
        gen_timer.stop(num_generated_tokens)
        hypos = [h[:args.nbest] for h in hypos]
        hypos = move_to_cpu(hypos) if use_cuda else hypos
        data_queue.put((cpu_sample, hypos))
        wps_meter.update(num_generated_tokens)
        progress.log({'wps': round(wps_meter.avg)})
        num_sentences += (
            cpu_sample['nsentences'] if "nsentences" in cpu_sample else cpu_sample["id"].numel()
        )
    
    data_queue.put(GENERATE_FINISHED)
    for p in p_list:
        p.join()
    message_queue.put(GENERATE_FINISHED)
    io_process.join()
    sent_through = num_sentences / gen_timer.sum if num_sentences > 0 else 0
    tokens_through = 1. / gen_timer.avg if num_sentences > 0 else 0
    logger.info("NOTE: hypothesis and token scores are output in base 2")
    logger.info(
        "Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)".format(
            num_sentences,
            gen_timer.n,
            gen_timer.sum,
            sent_through,
            tokens_through,
        )
    )
    return

@replace(main)
def main_v1(args):
    assert args.path is not None, '--path required for generation!'
    assert (
        not args.sampling or args.nbest == args.beam
    ), "--sampling requires --nbest to be equal to --beam"
    assert (
        args.replace_unk is None or args.dataset_impl == "raw"
    ), "--replace-unk requires a raw text dataset (--dataset-impl=raw)"
    
    if args.results_path is not None:
        os.makedirs(args.results_path, exist_ok = True)
        output_path = os.path.join(
            args.results_path, "generate-{}.txt".format(args.gen_subset)
        )
        with open(output_path, "w", buffering = 1, encoding = "utf-8") as h:
            return _main(args, h)
    else:
        return _main(args, sys.stdout)

