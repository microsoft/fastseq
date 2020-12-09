"""From Huggingface Transformers."""
import argparse
import json
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm

from fastseq_cli.transformers_utils import use_task_specific_params, trim_batch, calculate_rouge, calculate_bleu_score
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def sort_sentences(sents: List[str], reverse: bool=False):
    """Sort the input sentences by their length.

    Args:
        sents (List[str): input sentences.
        reverse (bool): indicate the order is ascending(False) or descending.

    Returns:
        tuple(List[str, List[int]): the sorted sentences and
            the indices in the original input list.
    """
    is_ascending = -1 if reverse else 1
    sorted_idx = sorted(
        range(len(sents)), key=lambda i: len(sents[i])*is_ascending)
    sorted_sents = [sents[i] for i in sorted_idx]
    return sorted_sents, sorted_idx

def unsort_sentences(sents: List[str], sorted_idx: List[int]):
    """Unsort the sents to be the order specified by sorted_idx.

    Args:
        sents (List[str]): a list of input strings.
        sorted_idx (List[int]): the order that will be restored.

    Returns:
        List[str]: the unsorted list of strings.
    """
    result = [''] * len(sents)
    for cur_idx, org_idx in enumerate(sorted_idx):
        result[org_idx] = sents[cur_idx]
    return result

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
    **gen_kwargs,
) -> None:
    """Run generation"""
    if fastseq_opt:
        import fastseq  #pylint: disable=import-outside-toplevel
        examples, sorted_idx = sort_sentences(examples, reverse=True)

    fout = Path(out_file).open("w", encoding="utf-8")
    model_name = str(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    if fp16:
        model = model.half()
    if decoder_start_token_id is None:
        decoder_start_token_id = gen_kwargs.pop("decoder_start_token_id", None)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # update config with summarization specific params
    use_task_specific_params(model, task)

    hypothesis = []
    for batch in tqdm(list(chunks(examples, batch_size))):
        if "t5" in model_name:
            batch = [model.config.prefix + text for text in batch]
        batch = tokenizer(batch,
                          return_tensors="pt",
                          truncation=True,
                          padding="max_length").to(device)
        input_ids, attention_mask = trim_batch(
            **batch, pad_token_id=tokenizer.pad_token_id)
        summaries = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_start_token_id=decoder_start_token_id,
            no_repeat_ngram_size=no_repeat_ngram_size,
            **gen_kwargs,
        )
        dec = tokenizer.batch_decode(summaries,
                                     skip_special_tokens=True,
                                     clean_up_tokenization_spaces=False)
        hypothesis.extend(dec)

    if fastseq_opt:
        hypothesis = unsort_sentences(hypothesis, sorted_idx)

    for hypo in hypothesis:
        fout.write(hypo + "\n")
        fout.flush()


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
