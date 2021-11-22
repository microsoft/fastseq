"""Run Huggingface Transformers as the baseline"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from rouge_score import rouge_scorer, scoring
from sacrebleu import corpus_bleu
import torch
from tqdm import tqdm

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM


DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ROUGE_KEYS = ["rouge1", "rouge2", "rougeL"]


def calculate_rouge(output_lns: List[str], reference_lns: List[str]) -> Dict:
    scorer = rouge_scorer.RougeScorer(ROUGE_KEYS, use_stemmer=True)
    aggregator = scoring.BootstrapAggregator()

    for reference_ln, output_ln in zip(reference_lns, output_lns):
        scores = scorer.score(reference_ln, output_ln)
        aggregator.add_scores(scores)

    result = aggregator.aggregate()
    return {k: v.mid.fmeasure for k, v in result.items()}

def use_task_specific_params(model, task):
    # update config with summarization specific params
    task_specific_params = model.config.task_specific_params
    if task_specific_params is not None:
        model.config.update(task_specific_params.get(task, {}))

def calculate_bleu_score(output_lns, refs_lns, **kwargs) -> dict:
    """Uses sacrebleu's corpus_bleu implementation."""
    return {"bleu": corpus_bleu(output_lns, [refs_lns], **kwargs).score}

def trim_batch(
    input_ids, pad_token_id, attention_mask=None, **kwargs
):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (
            input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]

def generate_summaries_or_translations(
    examples: list,
    out_file: str,
    model_name: str,
    batch_size: int = 8,
    device: str = DEFAULT_DEVICE,
    fp16=False,
    task="summarization",
    no_repeat_ngram_size=None,
    max_tokenizer_length=None,
    max_gen_length=None,
    **gen_kwargs,
) -> None:
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

    # update config with summarization specific params
    use_task_specific_params(model, task)

    for batch in tqdm(list(chunks(examples, batch_size))):
        if "t5" in model_name:
            batch = [model.config.prefix + text for text in batch]

        batch = tokenizer(
            batch,
            max_length=max_tokenizer_length,
            return_tensors="pt",
            truncation=True,
            padding="max_length").to(device)
        input_ids, attention_mask = trim_batch(
            **batch, pad_token_id=tokenizer.pad_token_id)
        summaries = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            no_repeat_ngram_size=no_repeat_ngram_size,
            max_length=max_gen_length,
            **gen_kwargs)
        dec = tokenizer.batch_decode(
            summaries, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)
        for hypothesis in dec:
            hypothesis = hypothesis.replace('\n', ' ')
            fout.write(hypothesis + "\n")
            fout.flush()


def run_generate():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str,
                        help="like facebook/bart-large-cnn,t5-base, etc.")
    parser.add_argument("input_path", type=str, help="like cnn_dm/test.source")
    parser.add_argument("save_path", type=str, help="where to save summaries")

    parser.add_argument("--reference_path", type=str, required=False,
                        help="like cnn_dm/test_reference_summaries.txt")
    parser.add_argument("--score_path", type=str, required=False,
                         help="where to save the rouge score in json format")
    parser.add_argument("--device", type=str, required=False,
                        default=DEFAULT_DEVICE, help="cuda, cuda:1, cpu etc.")
    parser.add_argument("--task", type=str, default="summarization",
                        help="typically translation or summarization")
    parser.add_argument("--bs", type=int, default=8, required=False,
                        help="batch size")
    parser.add_argument("--n_obs", type=int, default=-1, required=False,
                        help="How many observations. Defaults to all.")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=None,
                         required=False, help="size of no repeat ngram")
    parser.add_argument("--max_tokenizer_length", type=int,
                        help="max length for the tokenized sentence",
                        default=None, required=False)
    parser.add_argument("--max_gen_length", type=int,
                        help="max length for generation",
                        default=None, required=False)
    args = parser.parse_args()
    examples = [" " + x.rstrip() if "t5" in args.model_name else x.rstrip()
                for x in open(args.input_path).readlines()]
    if args.n_obs > 0:
        examples = examples[: args.n_obs]

    generate_summaries_or_translations(
        examples,
        args.save_path,
        args.model_name,
        batch_size=args.bs,
        device=args.device,
        fp16=args.fp16,
        task=args.task,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_tokenizer_length=args.max_tokenizer_length,
        max_gen_length=args.max_gen_length,
    )
    if args.reference_path is None:
        return
    # Compute scores
    score_fn = (calculate_bleu_score if "translation" in args.task
                else calculate_rouge)
    output_lns = [x.rstrip() for x in open(args.save_path).readlines()]
    reference_lns = [x.rstrip() for x in open(args.reference_path).readlines()
        ][: len(output_lns)]
    scores: dict = score_fn(output_lns, reference_lns)
    if args.score_path is not None:
        json.dump(scores, open(args.score_path, "w+"))
    return scores


if __name__ == "__main__":
    run_generate()
