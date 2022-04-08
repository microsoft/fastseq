"""From Huggingface Transformers."""

import sys
import logging
import argparse
import json
from pathlib import Path
from multiprocessing import Process, Queue
from tqdm import tqdm
import torch
from transformers import (AutoModelForSeq2SeqLM, AutoTokenizer,
                          AutoModelForCausalLM)
from fastseq_cli.transformers_utils import (
    use_task_specific_params, trim_batch, calculate_rouge, calculate_bleu_score)

from transformers.generation_utils import (
    BeamSearchEncoderDecoderOutput,
    BeamSearchDecoderOnlyOutput,
    GreedySearchDecoderOnlyOutput,
    GreedySearchEncoderDecoderOutput,
    SampleDecoderOnlyOutput,
    SampleEncoderDecoderOutput,
    BeamSampleDecoderOnlyOutput,
    BeamSampleEncoderDecoderOutput,
)

logger = logging.getLogger(__name__)

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

GENERATE_FINISHED = 'done'
POSTPROCESS_FINISHED = None

class TokenizeDataset(torch.utils.data.Dataset):
    """Characterizes a dataset for PyTorch"""
    def __init__(self, examples, tokenizer, model_name, prefix,
                return_tensors, truncation, padding, max_length=None):
        """Multiprocess Dataloader.
        Args:
            examples (List(str)): a list of input sentences.
            tokenizer (AutoTokenizer): instance of AutoTokenizer.
            model_name (string): model name.
            prefix (string): input example prefix if any.
        """
        self.examples = examples
        self.tokenizer= tokenizer
        self.model_name = model_name
        self.prefix = prefix
        self.return_tensors=return_tensors
        self.truncation=truncation
        self.padding=padding
        self.max_length=max_length

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        batch = self.examples[index]
        if "t5" in self.model_name:
            batch = self.prefix + batch
        batch = self.tokenizer(
            batch,
            max_length=self.max_length,
            return_tensors=self.return_tensors,
            truncation=self.truncation,
            padding=self.padding)
        return batch['input_ids'], batch['attention_mask']

class IOProcess (Process):
    """ Write detokenized output to file in order."""
    def __init__(self, msg_queue, fout):
        """Async output writer.
        Args:
            msg_queue : Multiprocess message Queue
            fout : output file pointer.
        """
        super(IOProcess, self).__init__()
        self.msg_queue = msg_queue
        self.fout = fout
        self.waiting_for=0
        self.dec_buf = {}

    def process_dec(self, dec_scores):
        """ Process and write detokenized hypotheses and scores
        Args:
            dec_scores (tuple(Tensor, Tensor or List)): tuple of (hypotheses, scores)
        """
        dec, scores = dec_scores
        for i, hypothesis in enumerate(dec):
            score = ''
            if scores is not None:
                score = scores[i] + '\t'
            hypothesis = hypothesis.replace('\n', ' ')
            self.fout.write(score + hypothesis + "\n")
            self.fout.flush()

    def process_buffer(self):
        while self.waiting_for in self.dec_buf:
            self.process_dec(self.dec_buf[self.waiting_for])
            del self.dec_buf[self.waiting_for]
            self.waiting_for+=1

    def run(self):
        while True:
            ind, dec, scores = self.msg_queue.get()
            if dec == GENERATE_FINISHED:
                break
            elif ind != self.waiting_for:
                self.dec_buf[ind] = (dec, scores)
            else:
                self.process_dec((dec, scores))
                self.waiting_for+=1
                self.process_buffer()
        self.process_buffer()
        assert not self.dec_buf, "IO Buffer not empty"
        self.msg_queue.close()
        self.msg_queue.join_thread()

class PostProcess(Process):
    """ Parallel detokenization """
    def __init__(self, tokenizer, data_queue, msg_queue,
            skip_special_tokens, clean_up_tokenization_spaces):
        """Async Postprocess.
        Args:
            data_queue : Multiprocess data Queue
            msg_queue :  Multiprocess message queue
            tokenizer : tokenizer
            clean_up_tokenization_spaces : clean_up_tokenization_spaces?
            skip_special_tokens = skip_special_tokens?
        """
        super(PostProcess, self).__init__()
        self.data_queue = data_queue
        self.msg_queue  = msg_queue
        self.tokenizer = tokenizer
        self.clean_up_tokenization_spaces = clean_up_tokenization_spaces
        self.skip_special_tokens = skip_special_tokens
        self.delimeter = "####"

    def run(self):
        while True:
            ind, summaries, scores = self.data_queue.get()
            if summaries == GENERATE_FINISHED:
                self.data_queue.put((-1, POSTPROCESS_FINISHED, None))
                break
            elif summaries == POSTPROCESS_FINISHED:
                self.data_queue.put((-1, POSTPROCESS_FINISHED, None))
                break
            else:
                no_scores = scores is not None and torch.all(torch.isnan(scores))
                if (len(summaries.shape) == 3):
                    bsz, num_ret_seq, seq_len = summaries.shape
                    dec = []
                    new_scores = []
                    for i in range(bsz):
                        current = summaries[i,:,:].reshape([num_ret_seq, seq_len])
                        cur_dec = []
                        for j in range(num_ret_seq):
                            cur_dec += [self.tokenizer.decode(summaries[i,j,:],
                                                              skip_special_tokens=self.skip_special_tokens,
                                                              clean_up_tokenization_spaces=self.clean_up_tokenization_spaces).strip()]
                        dec += [self.delimeter.join(cur_dec)]
                        if scores is not None:
                            current_scores = ""
                            for s in range(num_ret_seq):
                                x = scores[i][s]
                                if no_scores:
                                    x = 'NA'
                                current_scores += str(x) 
                                if s < num_ret_seq - 1:
                                    current_scores += self.delimeter
                            new_scores += [current_scores]
                    if scores is not None:
                        scores = new_scores
                else:
                    assert len(summaries.shape) == 2, "Summaries must have 2 or 3 dimensions"
                    dec = self.tokenizer.batch_decode(summaries,
                                                      skip_special_tokens=self.skip_special_tokens,
                                                      clean_up_tokenization_spaces=self.clean_up_tokenization_spaces)
                    if no_scores:
                        scores = ['NA'] * len(scores)
                    elif scores is not None:
                        scores = ["%f\t" %s.item() for s in scores]
                self.msg_queue.put((ind, dec, scores))
        self.data_queue.close()
        self.data_queue.join_thread()
        self.msg_queue.close()
        self.msg_queue.join_thread()

def generate_summaries_or_translations_baseline(
    examples: list,
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    decoder_start_token_id=None,
    no_repeat_ngram_size=None,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
    preprocess_workers=2,
    postprocess_workers=2,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_tokenizer_length=None,
    max_gen_length=None,
    max_new_tokens=None,
    use_causal_lm=False,
    output_summaries_only=False,
    output_sequence_scores=False,
    num_beams=None,
    eos_token_id=None,
    temperature=None,
    top_k=None,
    top_p=None,
    do_sample=None,
    repetition_penalty=None,
    num_return_sequences=None,
    padding_side=None,
    use_slow_tokenizer=False,
    **gen_kwargs,
) -> None:
    """Run generation"""
    assert 'fastseq' not in sys.modules, "Running with --without_fastseq_opt, Fastseq should not be imported."
    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)
    if use_causal_lm:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = not use_slow_tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device) 
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = not use_slow_tokenizer)

    if fp16:
        model = model.half()
    if decoder_start_token_id is None:
        decoder_start_token_id = gen_kwargs.pop("decoder_start_token_id", None)
    if hasattr(tokenizer, 'model_max_length') and max_tokenizer_length is not None:
        tokenizer.model_max_length = max_tokenizer_length
    if padding_side is not None:
        tokenizer.padding_side = padding_side

    # update config with summarization specific params
    use_task_specific_params(model, task)

    dataset = TokenizeDataset(examples, tokenizer, model_name,
        model.config.prefix, return_tensors, truncation, padding,
        max_tokenizer_length)
    training_generator = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, num_workers = 0,
            drop_last=False)
    try:
        for ind, batch in tqdm(enumerate(training_generator)):
            input_ids, attention_mask = batch
            input_ids = input_ids.view(input_ids.size(0), -1).to(device)
            attention_mask = attention_mask.view(input_ids.size(0), -1).to(device)
            input_ids, attention_mask = trim_batch(
              input_ids, tokenizer.pad_token_id, attention_mask)
            try:
                summaries = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=decoder_start_token_id,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    max_length=max_gen_length,
                    max_new_tokens=max_new_tokens,
                    output_scores=output_sequence_scores,
                    return_dict_in_generate=output_sequence_scores,
                    num_beams=num_beams,
                    eos_token_id=eos_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_return_sequences,
                    **gen_kwargs,
                )
            except:
                logger.exception(sys.exc_info()[0])
                sys.exit(1)
            if output_sequence_scores:
                sequences = summaries.sequences
            else:
                sequences = summaries
            scores_cpu = None
            if output_sequence_scores:
                if (type(summaries) in [BeamSearchEncoderDecoderOutput, 
                                        BeamSearchDecoderOnlyOutput, 
                                        BeamSampleDecoderOnlyOutput, 
                                        BeamSampleEncoderDecoderOutput]):
                        scores_cpu = summaries.sequences_scores.cpu()
                else: 
                    scores_cpu = ['NA'] * sequences.shape[0]
            if output_summaries_only:
                sequences = sequences[:, input_ids.shape[-1]:] 
            sequences_cpu = sequences.cpu()
            dec = tokenizer.batch_decode(
                sequences_cpu, skip_special_tokens=True,
                clean_up_tokenization_spaces=False)
            for i, hypothesis in enumerate(dec):
                hypothesis = hypothesis.replace('\n', ' ')
                score = ''
                if scores_cpu is not None:
                    if isinstance(scores_cpu[i], str):
                        score = scores_cpu[i] + '\t'
                    else:
                        score = "%f\t" %scores_cpu[i].item()
                fout.write(score + hypothesis + "\n")
                fout.flush()
    except:
        logger.exception(sys.exc_info()[0])
        sys.exit(1)
    fout.close()

def generate_summaries_or_translations_fast(
    examples: list,
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    decoder_start_token_id=None,
    no_repeat_ngram_size=None,
    skip_special_tokens=True,
    clean_up_tokenization_spaces=False,
    preprocess_workers=2,
    postprocess_workers=2,
    return_tensors="pt",
    truncation=True,
    padding="max_length",
    max_tokenizer_length=None,
    max_gen_length=None,
    max_new_tokens=None,
    use_causal_lm=False,
    output_summaries_only=False,
    output_sequence_scores=False,
    num_beams=None,
    eos_token_id=None,
    temperature=None,
    top_k=None,
    top_p=None,
    do_sample=None,
    repetition_penalty=None,
    num_return_sequences=None,
    padding_side=None,
    use_slow_tokenizer=False,
    **gen_kwargs,
) -> None:
    """Run generation"""
    import fastseq  #pylint: disable=import-outside-toplevel
    from fastseq.logging import get_logger #pylint: disable=import-outside-toplevel
    global logger
    logger = get_logger(__name__, logging.INFO)
    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)
    if use_causal_lm:
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = not use_slow_tokenizer)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast = not use_slow_tokenizer)

    if fp16:
        model = model.half()
    if decoder_start_token_id is None:
        decoder_start_token_id = gen_kwargs.pop("decoder_start_token_id", None)
    if hasattr(tokenizer, 'model_max_length') and max_tokenizer_length is not None:
        tokenizer.model_max_length = max_tokenizer_length
    if padding_side is not None:
        tokenizer.padding_side = padding_side

    # update config with summarization specific params
    use_task_specific_params(model, task)

    data_queue = Queue()
    msg_queue =  Queue()
    p_list = []

    for _ in range(postprocess_workers):
        p = PostProcess(tokenizer, data_queue, msg_queue,
            skip_special_tokens, clean_up_tokenization_spaces)
        p_list.append(p)
        p.start()

    io_process = IOProcess( msg_queue, fout)
    io_process.start()

    dataset = TokenizeDataset(examples, tokenizer, model_name,
        model.config.prefix, return_tensors, truncation, padding,
        max_tokenizer_length)
    training_generator = torch.utils.data.DataLoader(dataset,
            batch_size=batch_size, num_workers = preprocess_workers,
            drop_last=False)
    try:
        for ind, batch in tqdm(enumerate(training_generator)):
            input_ids, attention_mask = batch
            input_ids = input_ids.view(input_ids.size(0), -1).to(device)
            attention_mask = attention_mask.view(input_ids.size(0), -1).to(device)
            input_ids, attention_mask = trim_batch(
              input_ids, tokenizer.pad_token_id, attention_mask)
            try:
                summaries = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=decoder_start_token_id,
                    no_repeat_ngram_size=no_repeat_ngram_size,
                    max_length=max_gen_length,
                    max_new_tokens=max_new_tokens,
                    output_scores=output_sequence_scores,
                    return_dict_in_generate=output_sequence_scores,
                    num_beams=num_beams,
                    eos_token_id=eos_token_id,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    num_return_sequences=num_return_sequences,
                    **gen_kwargs,
                )
            except:
                logger.exception(sys.exc_info()[0])
                for p in p_list:
                    p.terminate()
                io_process.terminate()
                data_queue.close()
                msg_queue.close()
                sys.exit(1)
            if output_sequence_scores:
                sequences = summaries.sequences
            else:
                sequences = summaries
            scores_cpu = None
            if output_sequence_scores:
                if (type(summaries) in [BeamSearchEncoderDecoderOutput, 
                                        BeamSearchDecoderOnlyOutput, 
                                        BeamSampleDecoderOnlyOutput, 
                                        BeamSampleEncoderDecoderOutput]):
                        scores_cpu = summaries.sequences_scores.cpu()
                else: 
                    scores_cpu = torch.Tensor([float('nan')] * sequences.shape[0])
            if output_summaries_only:
                sequences = sequences[:, input_ids.shape[-1]:] 
            sequences_cpu = sequences.cpu()
            if (num_return_sequences is not None and num_return_sequences > 1):
                sequences_cpu = sequences_cpu.reshape([-1, num_return_sequences, sequences_cpu.shape[-1]])
                if (scores_cpu is not None):
                    scores_cpu = scores_cpu.reshape([-1, num_return_sequences])
            data_queue.put((ind, sequences_cpu, scores_cpu))
    except:
        logger.exception(sys.exc_info()[0])
        for p in p_list:
            p.terminate()
        io_process.terminate()
        data_queue.close()
        msg_queue.close()
        sys.exit(1)

    data_queue.put((-1, GENERATE_FINISHED, None))
    for p in p_list:
        p.join()
    msg_queue.put((-1, GENERATE_FINISHED, None))
    io_process.join()
    fout.close()

def run_generate():
    """Entrance is here."""
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",
                        type=str,
                        help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("save_path", type=str, help="where to save summaries")

    parser.add_argument("--reference_path",
                        type=str,
                        required=False,
                        help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument("--score_path",
                        type=str,
                        required=False,
                        help="where to save the rouge score in json format")
    parser.add_argument("--device",
                        type=str,
                        required=False,
                        default=DEFAULT_DEVICE,
                        help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--task",
                        type=str,
                        default="summarization",
                        help="typically translation or summarization")
    parser.add_argument("--bs",
                        type=int,
                        default=8,
                        required=False,
                        help="batch size")
    parser.add_argument(
        "--decoder_start_token_id",
        type=int,
        default=None,
        required=False,
        help="decoder_start_token_id (otherwise will look at config)",
    )
    parser.add_argument("--n_obs",
                        type=int,
                        default=-1,
                        required=False,
                        help="How many observations. Defaults to all.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--without_fastseq_opt", action="store_true")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None,
                         required=False, help="size of no repeat ngram")
    parser.add_argument("--include_special_tokens", action="store_true")
    parser.add_argument("--clean_up_tokenization_spaces", action="store_true")
    parser.add_argument("--preprocess_workers",
                        type=int,
                        default=2,
                        required=False,
                        help="pre-processing worker threads")
    parser.add_argument("--postprocess_workers",
                        type=int,
                        default=1,
                        required=False,
                        help="post-processing worker threads")
    parser.add_argument("--no_truncation", action="store_true")
    parser.add_argument("--return_tensors", type=str,
                        help="specify return tensors",
                        default="pt", required=False)
    parser.add_argument("--padding", type=str, help="specify padding",
                        default="max_length", required=False)
    parser.add_argument("--max_tokenizer_length", type=int,
                        help="max length for the tokenized sentence",
                        default=None, required=False)
    parser.add_argument("--max_gen_length", type=int,
                        help="max length for generation",
                        default=None, required=False),
    parser.add_argument("--max_new_tokens", type=int,
                        help="max tokens to generate, ignoring the number of "
                        "current tokens. Use either max_gen_length or "
                        "max_new_tokens, but not both - they serve the same purpose.",
                        default=None, required=False),
    parser.add_argument("--min_gen_length",
                        type=int,
                        default=-1,
                        required=False,
                        help="min length for decode")
    parser.add_argument("--causal_lm", action="store_true")
    parser.add_argument("--output_summaries_only", action="store_true")
    parser.add_argument("--output_sequence_scores", action="store_true")
    parser.add_argument("--beam",
                        type=int,
                        default=None,
                        required=False,
                        help="beam size for generation. If None, beam size will be loaded from the model configuration file (the parameter name is num_beams). If the model configuration file does not have this parameter, beam size will be set as 1")
    parser.add_argument("--eos_token_id", type=int,
                        default=None, required=False,
                        help="id fo the end-of-sequence token")
    parser.add_argument("--temperature", type=float,
                        default=None, required=False,
                        help="The value used to module the next token probabilities.")
    parser.add_argument("--top_k", type=int,
                        default=None, required=False,
                        help="The number of highest probability vocabulary tokens to "
                        "keep for top-k-filtering.")
    parser.add_argument("--top_p", type=float, 
                        default=None, required=False,
                        help="If set to float < 1, only the most probable tokens with "
                        "probabilities that add up to `top_p` or higher are kept for generation.")
    parser.add_argument("--repetition_penalty", type=float,
                        default=None, required=False,
                        help="The parameter for repetition penalty. 1.0 means no penalty.")
    parser.add_argument("--do_sample", action="store_true",
                        help="Whether or not to use sampling ; use greedy decoding otherwise.")
    parser.add_argument("--num_return_sequences", type=int,
                        default=None, required=False, 
                        help="The number of independently computed returned sequences for each element in the batch.")
    parser.add_argument("--seed", type=int, default=None, required=False,
                        help="Specify a random seed for initialization")
    parser.add_argument("--padding_side", type=str, default=None, required=False,
                        help="Specify which side the tokenizer should pad")
    parser.add_argument("--use_slow_tokenizer", action="store_true",
                        help="Try to load regular <model>Tokenizer instead of <model>TokenizerFast (default)")
    args = parser.parse_args()
    examples = [
        " " + x.rstrip() if "t5" in args.model_name else x.rstrip()
        for x in open(args.input_path).readlines()
    ]
    if args.n_obs > 0:
        examples = examples[:args.n_obs]
    Path(args.save_path).parent.mkdir(exist_ok=True)
    if args.seed is not None:
        torch.manual_seed(args.seed)
    if args.without_fastseq_opt:
        generate_summaries_or_translations_baseline(
            examples,
            args.save_path,
            args.model_name,
            batch_size=args.bs,
            device=args.device,
            fp16=args.fp16,
            task=args.task,
            decoder_start_token_id=args.decoder_start_token_id,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            skip_special_tokens=not args.include_special_tokens,
            clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
            preprocess_workers=args.preprocess_workers,
            postprocess_workers=args.postprocess_workers,
            return_tensors=args.return_tensors,
            truncation=not args.no_truncation,
            padding=args.padding,
            max_tokenizer_length=args.max_tokenizer_length,
            max_gen_length=args.max_gen_length,
            max_new_tokens=args.max_new_tokens,
            use_causal_lm=args.causal_lm,
            output_summaries_only=args.output_summaries_only,
            output_sequence_scores=args.output_sequence_scores,
            num_beams=args.beam,
            eos_token_id=args.eos_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
            padding_side=args.padding_side,
            use_slow_tokenizer=args.use_slow_tokenizer,
            )
    else:
        generate_summaries_or_translations_fast(
            examples,
            args.save_path,
            args.model_name,
            batch_size=args.bs,
            device=args.device,
            fp16=args.fp16,
            task=args.task,
            decoder_start_token_id=args.decoder_start_token_id,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            skip_special_tokens=not args.include_special_tokens,
            clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
            preprocess_workers=args.preprocess_workers,
            postprocess_workers=args.postprocess_workers,
            return_tensors=args.return_tensors,
            truncation=not args.no_truncation,
            padding=args.padding,
            max_tokenizer_length=args.max_tokenizer_length,
            max_gen_length=args.max_gen_length,
            max_new_tokens=args.max_new_tokens,
            use_causal_lm=args.causal_lm,
            output_summaries_only=args.output_summaries_only,
            output_sequence_scores=args.output_sequence_scores,
            num_beams=args.beam,
            eos_token_id=args.eos_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            do_sample=args.do_sample,
            num_return_sequences=args.num_return_sequences,
            padding_side=args.padding_side,
            use_slow_tokenizer=args.use_slow_tokenizer,
            )

    if args.reference_path is None:
        return
    # Compute scores
    score_fn = calculate_bleu_score \
        if "translation" in args.task else calculate_rouge
    output_lns = [x.rstrip() for x in open(args.save_path).readlines()]
    reference_lns = [
        x.rstrip() for x in open(args.reference_path).readlines()
    ][:len(output_lns)]
    scores: dict = score_fn(output_lns, reference_lns)
    if args.score_path is not None:
        json.dump(scores, open(args.score_path, "w+"))
    return


if __name__ == "__main__":
    run_generate()
