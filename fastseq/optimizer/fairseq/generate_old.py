# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Optimize fairseq-generate (v0.9.0)"""

import logging
import sys
from multiprocessing import Queue, JoinableQueue
from torch.multiprocessing import Process

import torch

from fairseq_cli.generate import main
from fairseq import bleu, checkpoint_utils, options, progress_bar, tasks, utils
from fairseq.data import encoders
from fairseq.meters import StopwatchMeter, TimeMeter
from fairseq.options import add_generation_args
from fairseq.utils import apply_to_sample

from fastseq.utils.api_decorator import replace

GENERATE_FINISHED = "done"
POSTPROCESS_FINISHED = None

def move_to_cpu(sample):
    def _move_to_cpu(tensor):
        # PyTorch has poor support for half tensors (float16) on CPU.
        # Move any such tensors to float32.
        if tensor.dtype in {torch.bfloat16, torch.float16}:
            return tensor.cpu().to(dtype=torch.float32)
        else:
            return tensor.cpu()

    return apply_to_sample(_move_to_cpu, sample)


class IOProcess(Process):
    """
    Single process to hanlde IO and compute metrics
    """
    def __init__(self, args, task, message_queue):
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
        if args.sacrebleu:
            self.scorer = bleu.SacrebleuScorer()
        else:
            self.scorer = bleu.Scorer(self.tgt_dict.pad(), self.tgt_dict.eos(),
                                      self.tgt_dict.unk())

        self.args = args
        self.message_queue = message_queue
        self.has_target = False

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
                    print('| Generate {} with beam={}: {}'.format(
                        self.args.gen_subset, self.args.beam,
                        self.scorer.result_string()))
                break
            else:
                print(msg)
            self.message_queue.task_done()
        self.message_queue.close()
        self.message_queue.join_thread()


class PostProcess(Process):
    '''
    Use multiple process to do detokenize
    '''

    def __init__(self, args, task, data_queue, message_queue):
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
        if args.sacrebleu:
            self.scorer = bleu.SacrebleuScorer()
        else:
            self.scorer = bleu.Scorer(self.tgt_dict.pad(), self.tgt_dict.eos(),
                                      self.tgt_dict.unk())

        self.args = args
        self.task = task
        self.data_queue = data_queue
        self.message_queue = message_queue
        if args.decode_hypothesis:
            self.tokenizer = encoders.build_tokenizer(args)
            self.bpe = encoders.build_bpe(args)

    def _decode(self, x):
        if self.bpe is not None:
            x = self.bpe.decode(x)
        if self.tokenizer is not None:
            x = self.tokenizer.decode(x)
        return x

    def _detokenize(self, sample, hypos):
        """ detokenize and compute BELU """
        message_list = []
        for i, sample_id in enumerate(sample['id'].tolist()):
            has_target = sample['target'] is not None

            # Remove padding
            src_tokens = utils.strip_pad(
                sample['net_input']['src_tokens'][i, :], self.tgt_dict.pad())
            target_tokens = None
            if has_target:
                target_tokens = utils.strip_pad(sample['target'][i, :],
                                                self.tgt_dict.pad()).int()

            # Either retrieve the original sentences or regenerate them from tokens.
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
                        target_tokens, self.args.remove_bpe, escape_unk=True)

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
                hypo_tokens, hypo_str, alignment = \
                    utils.post_process_prediction(
                        hypo_tokens=hypo['tokens'].int(),
                        src_str=src_str,
                        alignment=hypo['alignment'],
                        align_dict=self.align_dict,
                        tgt_dict=self.tgt_dict,
                        remove_bpe=self.args.remove_bpe,
                    )

                if not self.args.quiet:
                    if self.args.decode_hypothesis:
                        detok_hypo_str = self._decode(hypo_str)
                        message_list.append('D-{}\t{}\t{}'.format(
                            sample_id, hypo['score'], detok_hypo_str))
                    else:
                        message_list.append('H-{}\t{}\t{}'.format(
                            sample_id, hypo['score'], hypo_str))
                    message_list.append('P-{}\t{}'.format(
                        sample_id, ' '.join(
                            map(
                                lambda x: '{:.4f}'.format(x),
                                hypo['positional_scores'].tolist(),
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
                        message_list.append("\n".join([
                            'E-{}_{}\t{}'.format(sample_id, step,
                                                 utils.post_process_prediction(
                                                     h['tokens'].int(),
                                                     self.src_str, None, None,
                                                     self.tgt_dict, None)[1])
                            for step, h in enumerate(hypo['history'])
                        ]))

                # Score only the top hypothesis
                if has_target and j == 0:
                    if (self.align_dict is not None or
                        self.args.remove_bpe is not None):
                        # Convert back to tokens for evaluation with unk
                        # replacement and/or without BPE
                        target_tokens = self.tgt_dict.encode_line(
                            target_str, add_if_not_exist=True)
                    if hasattr(self.scorer, 'add_string'):
                        self.message_queue.put((target_str, hypo_str))
                    else:
                        self.message_queue.put((target_tokens, hypo_tokens))

        self.message_queue.put('\n'.join(message_list))

    def run(self):
        while True:
            r = self.data_queue.get()
            if r == GENERATE_FINISHED:
                self.data_queue.put(POSTPROCESS_FINISHED)
                break
            elif r is POSTPROCESS_FINISHED:
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
        action="store_true")
    # fmt: on


# @replace(main)
def main_v1(args):
    assert args.path is not None, '--path required for generation!'
    assert not args.sampling or args.nbest == args.beam, \
        '--sampling requires --nbest to be equal to --beam'
    assert args.replace_unk is None or args.raw_text, \
        '--replace-unk requires a raw text dataset (--raw-text)'

    utils.import_user_module(args)

    if args.max_tokens is None and args.max_sentences is None:
        args.max_tokens = 12000
    print(args)

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
    print('| loading model(s) from {}'.format(args.path))
    models, model_args_ = checkpoint_utils.load_model_ensemble(
        args.path.split(':'),
        arg_overrides=eval(args.model_overrides),
        task=task,
    )

    # Optimize ensemble for generation
    for model in models:
        model.make_generation_fast_(
            beamable_mm_beam_size=None if args.no_beamable_mm else args.beam,
            need_attn=args.print_alignment,
        )
        if args.fp16:
            model.half()
        if use_cuda:
            model.cuda()

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

    # Initialize generator
    gen_timer = StopwatchMeter()
    generator = task.build_generator(args)

    num_sentences = 0
    data_queue = Queue()
    message_queue = JoinableQueue()

    p_list = []
    for i in range(args.postprocess_workers):
        p = PostProcess(args, task, data_queue, message_queue)
        p_list.append(p)
        p.start()

    io_process = IOProcess(args, task, message_queue)
    io_process.start()

    if args.use_el_attn:
        task.transpose_enc_dec_kv_proj(models)
    with progress_bar.build_progress_bar(args, itr) as t:
        wps_meter = TimeMeter()
        for sample in t:
            cpu_sample = sample
            if 'net_input' not in sample:
                continue
            sample = utils.move_to_cuda(sample) if use_cuda else sample

            prefix_tokens = None
            if args.prefix_size > 0:
                prefix_tokens = sample['target'][:, :args.prefix_size]

            gen_timer.start()
            try:
                hypos = task.inference_step(
                    generator, models, sample, prefix_tokens)
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
            t.log({'wps': round(wps_meter.avg)})
            num_sentences += cpu_sample['nsentences']

    data_queue.put(GENERATE_FINISHED)
    for p in p_list:
        p.join()

    sent_throught = num_sentences / gen_timer.sum if num_sentences > 0 else 0
    tokens_throught = 1. / gen_timer.avg if num_sentences > 0 else 0

    message_queue.put(
        '| Translated {} sentences ({} tokens) in {:.1f}s ({:.2f} sentences/s, {:.2f} tokens/s)'. # pylint: disable=line-too-long
        format(num_sentences, gen_timer.n, gen_timer.sum, sent_throught,
               tokens_throught))

    message_queue.put(GENERATE_FINISHED)
    io_process.join()

    return
