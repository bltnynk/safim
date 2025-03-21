import argparse
# from human_eval_infilling.data import read_problems, stream_jsonl, write_jsonl
import copy
from tqdm import tqdm
from typing import Dict, Iterable
import gzip
from data_utils import load_dataset
import json
import os

BENCHMARK_NAME = ["single-line", "multi-line", "random-span", "random-span-light"]
STOP_WORDS = ["<|endoftext|>", "<|filename|>", "<file_sep>"]
COMPLETION_PLACEHOLDER = {
    "python": "# TODO: Your code here",
    "java": "/* TODO: Your code here */",
    "cpp": "/* TODO: Your code here */",
    "csharp": "/* TODO: Your code here */",
}

def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    """
    if append:
        mode = "ab"
    else:
        mode = "wb"
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode="wb") as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode("utf-8"))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode("utf-8"))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate HumanEval Task Inputs",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("completion_type", type=str)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Number of samples to solve and evaluate from the benchmark",
    )
    parser.add_argument(
        "--n_copies",
        type=int,
        default=1,
        help="Number of samples to solve and evaluate from the benchmark",
    )

    args = parser.parse_args()
    
    return args

def assemble_infilling_prompt(prefix: str, suffix: str, reverse: bool = False) -> str:
        if reverse:
            return "<fim_prefix>" + "<fim_suffix>" + suffix + "<fim_middle>" + prefix
        else:
            return "<fim_prefix>" + prefix + "<fim_suffix>" + suffix + "<fim_middle>"

def get_infilling_parts(sample):
    parts = sample["prompt"].split(COMPLETION_PLACEHOLDER[sample["lang"]])
    assert len(parts) == 2
    return parts

def apply_prompt(
    sample: dict,
) -> str:
    prefix, suffix = get_infilling_parts(sample)
    prompt = assemble_infilling_prompt(prefix, suffix, reverse=False)
    return prompt

if __name__ == "__main__":
    args = parse_args()
    inputs = []
    task_name = "safim"
    for i, sample in enumerate(tqdm(load_dataset(args.completion_type))): # block, control, api
        if args.limit and i >= args.limit:
            break
        for copy_ind in range(args.n_copies):
            input = copy.deepcopy(sample)
            input['task_name'] = task_name
            input['id'] = input['task_id'] + f"_copy{copy_ind}"
            input['copy'] = copy_ind
            input['text'] = apply_prompt(sample)
            input['stop_words'] = STOP_WORDS
            inputs.append(input)
    write_jsonl(f"data/SAFIM_{args.completion_type}-Inputs.json", inputs)
    print(f'Wrote tasks to data/SAFIM_{args.completion_type}-Inputs.json')
