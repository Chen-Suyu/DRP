import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import argparse
import time
import warnings

import evaluate
import torch.distributed as dist
from datasets import load_from_disk
from transformers import set_seed
from tqdm import tqdm
from utils.utils import write_to_csv
import pandas as pd


warnings.filterwarnings("ignore")
set_seed(42)
if dist.is_initialized():
    dist.barrier()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Evaluate generated reviews")

    parser.add_argument("--method", default="DRP", help="Generation method")

    parser.add_argument(
        "--diff_model_name",
        default="Qwen/Qwen2.5-14B-Instruct",
        choices=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        ],
        help="Model identifier",
    )

    parser.add_argument("--dataset", choices=["val", "test"], default="test")

    parser.add_argument(
        "--category",
        choices=["Movies_and_TV", "CDs_and_Vinyl", "Books"],
        default="Books",
    )

    parser.add_argument("--num_documents", type=int, default=0)
    parser.add_argument("--num_retrieved", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--num_users", type=int, default=4)

    parser.add_argument("--output_dir", type=str, default="./output")

    return parser.parse_args()


def load_predictions(predictions_path):
    if not os.path.exists(predictions_path):
        raise FileNotFoundError(f"Predictions file not found: {predictions_path}")
    with open(predictions_path, "r", encoding="utf-8") as f:
        raw = f.read().split("\n---------------------------------\n")[:-1]
        return [p for p in raw]


def evaluate_predictions(predictions, references):
    print("Loading evaluation metrics...")
    start = time.time()
    bleu = evaluate.load("sacrebleu")
    rouge = evaluate.load("rouge")
    meteor = evaluate.load("meteor")
    print(f"Metrics loaded in {time.time() - start:.2f}s\n")

    results = {}

    # BLEU
    print("Computing BLEU...")
    start = time.time()
    bleu_score = bleu.compute(predictions=predictions, references=references)["score"]
    print(f"BLEU computed in {time.time() - start:.2f}s")
    results["bleu"] = bleu_score

    # METEOR
    print("Computing METEOR...")
    start = time.time()
    meteor_score = meteor.compute(predictions=predictions, references=references)[
        "meteor"
    ]
    print(f"METEOR computed in {time.time() - start:.2f}s")
    results["meteor"] = meteor_score

    # ROUGE
    print("Computing ROUGE...")
    start = time.time()
    rouge_scores = rouge.compute(predictions=predictions, references=references)
    print(f"ROUGE computed in {time.time() - start:.2f}s")
    results["rouge-1"] = rouge_scores["rouge1"]
    results["rouge-L"] = rouge_scores["rougeL"]

    return results


def main():
    args = parse_arguments()

    output_dir = args.output_dir
    method = args.method
    num_documents = args.num_documents
    num_retrieved = args.num_retrieved
    num_users = args.num_users
    category = args.category

    diff_model_name = args.diff_model_name

    output_path = os.path.join(output_dir, method, category, "final_outputs")
    os.makedirs(output_path, exist_ok=True)

    load_name = f"{category}_doc{num_documents}_ret{num_retrieved}_user{num_users}_{diff_model_name.split('/')[1]}.txt"
    load_path = os.path.join(output_path, load_name)

    print(f"Loading predictions from: {load_path}")
    predictions = load_predictions(load_path)

    print("Loading ground-truth data...")
    dataset = load_from_disk(
        f"./data/DRP-main/{category}/{args.dataset}"
    )

    references = [sample["data"]["text"] for sample in dataset]

    print("Evaluating predictions...")
    results = evaluate_predictions(predictions, references)

    print("Evaluation Results:")
    for metric, score in results.items():
        print(f"{metric}: {score:.4f}")

    print("Writing results to CSV...")
    task_name = load_name[:-4]
    for metric, score in results.items():
        write_to_csv(task_name, metric, score)

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
