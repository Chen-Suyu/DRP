import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import argparse
import pickle
import warnings
from collections import defaultdict

from tqdm import tqdm
import torch.distributed as dist
from datasets import load_from_disk
from transformers import set_seed
from sentence_transformers import SentenceTransformer

from utils.preprocess import create_diff_inputs


warnings.filterwarnings("ignore")
set_seed(42)
if dist.is_initialized():
    dist.barrier()


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu_id", default="0")

    parser.add_argument("--method", default="DRP")
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

    parser.add_argument("--batch_size", type=int, default=8,
                    help="Batch size for embedding to avoid OOM")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    output_dir = args.output_dir
    method = args.method
    num_documents = args.num_documents
    num_retrieved = args.num_retrieved
    num_users = args.num_users
    category = args.category

    main_dataset = load_from_disk(
        f"./data/DRP-main/{category}/{args.dataset}"
    )
    meta_dataset = load_from_disk(
        f"./data/DRP-meta/{category}/full"
    )



    user_profile_map = {}
    asin_reviewers_map = defaultdict(set)
    for sample in main_dataset:
        user_id = sample["user_id"]
        user_profile_map[user_id] = sample["profile"]
        for p in sample["profile"]:
            asin_reviewers_map[p["asin"]].add(user_id)
    asin_map = dict(
        zip(
            meta_dataset["asin"],
            zip(meta_dataset["title"], meta_dataset["description"]),
        )
    )

    embedder = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct")

    diff_inputs = []
    for sample in tqdm(main_dataset, desc="Data-processing", total=len(main_dataset)):
        user_id = sample["user_id"]
        profile = user_profile_map[user_id]
        for p in profile:
            asin = p["asin"]
            item_title, description = asin_map[asin]
            p["item_title"] = item_title
            p["description"] = description

        data = sample["data"]
        asin = data["asin"]
        item_title, description = asin_map[asin]
        text, rating, review_title = data["text"], data["rating"], data["title"]
        diff_input = create_diff_inputs(
            num_documents,
            num_retrieved,
            num_users,
            user_id,
            profile,
            description,
            review_title,
            user_profile_map,
            asin_reviewers_map,
            embedder,
        )
        diff_inputs.append(diff_input)

    output_path = os.path.join(args.output_dir, args.method, category, "diff_inputs")
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(
        output_path,
        f"{category}_doc{num_documents}_ret{num_retrieved}_user{num_users}.pkl",
    )
    with open(save_path, "wb") as f:
        pickle.dump(diff_inputs, f)

    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)
