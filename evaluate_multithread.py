import concurrent.futures
import math
import argparse
import ast
import json
import re

import tqdm
from tqdm import tqdm
import numpy as np
from typing import List, Union
import itertools
from ast_utils import ErrorCheckVisitor, get_parser
from data_utils import load_dataset, stream_jsonl
from exec_utils import build_execeval, run_test

STOP_WORDS = ["<|endoftext|>", "<|filename|>", "<file_sep>"]
FIM_MIDDLE = '<fim_middle>'

def estimate_pass_at_k(
    num_samples: Union[int, List[int], np.ndarray],
    num_correct: Union[List[int], np.ndarray],
    k: int,
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.
    """

    def estimator(n: int, c: int, k: int) -> float:
        """
        Calculates 1 - comb(n - c, k) / comb(n, k).
        """
        if n - c < k:
            return 1.0
        return 1.0 - np.prod(1.0 - k / np.arange(n - c + 1, n + 1))

    if isinstance(num_samples, int):
        num_samples_it = itertools.repeat(num_samples, len(num_correct))
    else:
        assert len(num_samples) == len(num_correct)
        num_samples_it = iter(num_samples)

    return np.array([estimator(int(n), int(c), k) for n, c in zip(num_samples_it, num_correct)])

def check_syntax(code):
    parser = get_parser("python")
    code_bytes = code.encode("utf-8")
    tree = parser.parse(code_bytes)
    error_check = ErrorCheckVisitor()
    error_check(tree)
    return error_check.error_cnt == 0


def get_function_call_params(node):
    positional_args = [ast.dump(arg) for arg in node.args]
    keyword_args = {kw.arg: ast.dump(kw.value) for kw in node.keywords}
    return positional_args, keyword_args


def function_calls_match(call1, call2):
    params1 = get_function_call_params(call1)
    params2 = get_function_call_params(call2)
    return params1 == params2


def syntax_match(code1, code2, lang):
    code1 = re.sub(r'\s+', '', code1).strip()
    code2 = re.sub(r'\s+', '', code2).strip()
    if lang == "python":
        try:
            tree1 = ast.parse(code1, mode='eval')
            tree2 = ast.parse(code2, mode='eval')

            if isinstance(tree1.body, ast.Call) and isinstance(tree2.body, ast.Call):
                return function_calls_match(tree1.body, tree2.body)
        except:
            pass  # If parsing fails, fall back to simple string comparison

    return code1 == code2

import concurrent.futures
import math
from tqdm import tqdm

def process_problem_subset(problems, completions, progress_bar):
    partial_results = []
    pass_cnt = 0
    total = []
    correct = []
    for problem in problems:
        if problem["task_id"] not in completions:
            result = "EMPTY"
            passed = False
        else:
            total.append(len(completions[problem["task_id"]]))
            correct.append(0)
            for sample_id, completion in enumerate(completions[problem["task_id"]]):
                # completion = completions[problem["task_id"]]
                completion_output = completion["output"]
                completion_start = completion_output.find(FIM_MIDDLE) + len(FIM_MIDDLE)
                completion_end = len(completion_output)
                for stop_word in STOP_WORDS:
                    if stop_word in completion_output:
                        completion_end = min(completion_end, completion_output.find(stop_word))
                completion_output = completion_output[completion_start:completion_end]
                completion['completion'] = completion_output
                if "unit_tests" in problem and problem["unit_tests"]:
                    if completion['completion'] == problem["ground_truth"]:
                        result = "PASSED"
                        passed = True
                    else:
                        result, passed = run_test(problem, completion)
                else:
                    if syntax_match(completion['completion'], problem["ground_truth"], problem["lang"]):
                        result = "EXACT_MATCH"
                        passed = True
                    else:
                        result = "WRONG_ANSWER"
                        passed = False
                if not completion['completion'].strip() and not passed:
                    result = "EMPTY"
                if problem["lang"] == "python" and not passed:
                    full_code = problem['eval_prompt'].replace("{{completion}}", completion['completion'])
                    if "unit_tests" in problem and not check_syntax(full_code):
                        result = "COMPILATION_ERROR"
                pass_cnt += int(passed)
                correct[-1] += int(passed)
                partial_results.append(
                    {
                        "task_id": problem["task_id"], "copy": sample_id, "result": result, "passed": passed, "check_result": 0
                    }
                )
            progress_bar.update(1)  # Update progress bar after processing a problem
    
    return partial_results, pass_cnt, total, correct


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("completion_type", type=str)
    parser.add_argument("completion_path", type=str)
    # parser.add_argument("output_path", type=str)
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--num_threads", type=int, default=32)
    args = parser.parse_args()

    build_execeval(args)
    n_copies = 0
    # completions = {completion["task_id"]: completion for completion in stream_jsonl(args.completion_path)}
    completions = {}
    for completion in stream_jsonl(args.completion_path):
        if completion["task_id"] not in completions:
            completions[completion["task_id"]] = []
        completions[completion["task_id"]].append(completion)
        n_copies += 1
    problems = load_dataset(args.completion_type)
    total_problems = len(problems)
    assert n_copies % total_problems == 0, f"Number of completions {n_copies} is not divisible by number of problems {total_problems}"
    n_copies //= total_problems

    # Split the dataset into chunks for threads
    chunk_size = math.ceil(total_problems / args.num_threads)
    problem_chunks = [problems[i:i + chunk_size] for i in range(0, total_problems, chunk_size)]

    # Set up a shared progress bar
    with tqdm(total=total_problems, desc="Processing", unit="problem") as progress_bar:
        # Use ThreadPoolExecutor for multithreading
        results = []
        pass_cnt = 0
        total = []
        correct = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
            futures = {
                executor.submit(process_problem_subset, chunk, completions, progress_bar): chunk
                for chunk in problem_chunks
            }
            for future in concurrent.futures.as_completed(futures):
                partial_results, partial_pass_cnt, partial_total, partial_correct = future.result()
                results.extend(partial_results)
                pass_cnt += partial_pass_cnt
                total.extend(partial_total)
                correct.extend(partial_correct)

    total = np.array(total)
    correct = np.array(correct)
    ks = [1, 5, 10, 20, 50, 100]
    pass_at_k = {
        f"pass@{k}": estimate_pass_at_k(total, correct, k).mean() for k in ks if (total >= k).all()
    }
    # Aggregate and write final results
    print(f"Pass {pass_cnt} / Total {total_problems * n_copies}")
    print(f"Pass@1: {pass_cnt / (total_problems * n_copies) * 100 :.04f}%")
    output_path = args.completion_path[:args.completion_path.find("output")] + "results.jsonl"
    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    eval = {
        "pass": pass_cnt,
        "total": total_problems * n_copies,
        "total_problems": total_problems,
        "n_copies": n_copies,
        "pass@1": pass_cnt / total * 100,
        **pass_at_k
    }
    results_file = args.completion_path[:args.completion_path.find("output")] + "eval.jsonl"
    with open(results_file, "w") as f:
        f.write(json.dumps(eval, indent=2))
    print(f"Results saved to {results_file}")


if __name__ == '__main__':
    main()
