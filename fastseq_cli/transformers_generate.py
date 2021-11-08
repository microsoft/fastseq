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

    def process_dec(self, dec):
        for hypothesis in dec:
            hypothesis = hypothesis.replace('\n', ' ')
            self.fout.write(hypothesis + "\n")
            self.fout.flush()

    def process_buffer(self):
        while self.waiting_for in self.dec_buf:
            self.process_dec(self.dec_buf[self.waiting_for])
            del self.dec_buf[self.waiting_for]
            self.waiting_for+=1

    def run(self):
        while True:
            ind, dec = self.msg_queue.get()
            if dec == GENERATE_FINISHED:
                break
            elif ind != self.waiting_for:
                self.dec_buf[ind] = dec
            else:
                self.process_dec(dec)
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

    def run(self):
        while True:
            ind, summaries = self.data_queue.get()
            if summaries == GENERATE_FINISHED:
                self.data_queue.put((-1, POSTPROCESS_FINISHED))
                break
            elif summaries == POSTPROCESS_FINISHED:
                self.data_queue.put((-1, POSTPROCESS_FINISHED))
                break
            else:
                dec = self.tokenizer.batch_decode(summaries,
                        skip_special_tokens = self.skip_special_tokens,
                        clean_up_tokenization_spaces =
                        self.clean_up_tokenization_spaces)
                self.msg_queue.put((ind, dec))

        self.data_queue.close()
        self.data_queue.join_thread()
        self.msg_queue.close()
        self.msg_queue.join_thread()

def generate_summaries_or_translations(
    examples: list,
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    decoder_start_token_id=None,
    fastseq_opt=True,
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
    **gen_kwargs,
) -> None:
    """Run generation"""
    if fastseq_opt:
        import fastseq  #pylint: disable=import-outside-toplevel
    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)
    if model_name == 'gpt2':
        model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

    if fp16:
        model = model.half()
    if decoder_start_token_id is None:
        decoder_start_token_id = gen_kwargs.pop("decoder_start_token_id", None)

    if hasattr(tokenizer, 'model_max_length') and max_tokenizer_length is not None:
        tokenizer.model_max_length = max_tokenizer_length

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
                    **gen_kwargs,
                )
            except:
                logging.exception(sys.exc_info()[0])
                for p in p_list:
                    p.terminate()
                io_process.terminate()
                data_queue.close()
                msg_queue.close()
                sys.exit(1)
            summaries_cpu = summaries.cpu()
            data_queue.put((ind, summaries_cpu))
    except:
        logging.exception(sys.exc_info()[0])
        for p in p_list:
            p.terminate()
        io_process.terminate()
        data_queue.close()
        msg_queue.close()
        sys.exit(1)
    data_queue.put((-1, GENERATE_FINISHED))
    for p in p_list:
        p.join()
    msg_queue.put((-1, GENERATE_FINISHED))
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
                        default=None, required=False)
    parser.add_argument("--min_gen_length",
                        type=int,
                        default=-1,
                        required=False,
                        help="min length for decode")

    args = parser.parse_args()
    examples = [
        " " + x.rstrip() if "t5" in args.model_name else x.rstrip()
        for x in open(args.input_path).readlines()
    ]
    if args.n_obs > 0:
        examples = examples[:args.n_obs]
    Path(args.save_path).parent.mkdir(exist_ok=True)
    generate_summaries_or_translations(
        examples,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        decoder_start_token_id=args.decoder_start_token_id,
        fastseq_opt=not args.without_fastseq_opt,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        skip_special_tokens=not args.include_special_tokens,
        clean_up_tokenization_spaces=args.clean_up_tokenization_spaces,
        preprocess_workers=args.preprocess_workers,
        postprocess_workers=args.postprocess_workers,
        return_tensors=args.return_tensors,
        truncation=not args.no_truncation,
        padding=args.padding,
        max_tokenizer_length=args.max_tokenizer_length,
        max_gen_length=args.max_gen_length
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
    print(scores)
    if args.score_path is not None:
        json.dump(scores, open(args.score_path, "w+"))
    return


if __name__ == "__main__":
    run_generate()
