import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import argparse
import pickle
import warnings

import torch.distributed as dist
from datasets import load_from_disk
from transformers import set_seed
from utils.templates import DeepSeekR1PromptTemplate
from vllm import SamplingParams, LLM

from utils.get_local_model import get_local_model

warnings.filterwarnings("ignore")
set_seed(42)
if dist.is_initialized():
    dist.barrier()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", default="4,5")

    parser.add_argument("--method", default="DRP")
    parser.add_argument("--dataset", choices=["val", "test"], default="test")

    parser.add_argument(
        "--model_name",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        choices=[
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
            "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        ],
    )
    parser.add_argument("--max_tokens", type=int, default=1024 * 4)
    parser.add_argument("--temperature", type=float, default=0)

    parser.add_argument(
        "--category",
        choices=["Movies_and_TV", "CDs_and_Vinyl", "Books"],
        default="Books",
    )
    parser.add_argument("--num_documents", type=int, default=0)
    parser.add_argument("--num_retrieved", type=int, choices=range(1, 9), default=8)
    parser.add_argument("--num_users", type=int, default=4)

    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument(
        "--tag",
        default="DeepSeek-R1-Distill-Qwen-14B",
        choices=[
            "DeepSeek-R1-Distill-Qwen-1.5B",
            "DeepSeek-R1-Distill-Qwen-7B",
            "DeepSeek-R1-Distill-Qwen-14B",
            "DeepSeek-R1-Distill-Qwen-32B",
        ],
    )
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
    tag = args.tag

    main_dataset = load_from_disk(
        f"./data/DRP-main/{category}/{args.dataset}"
    )
    meta_dataset = load_from_disk(
        f"./data/DRP-meta/{category}/full"
    )

    load_path = os.path.join(
        args.output_dir,
        args.method,
        category,
        "diff_inputs",
        f"{category}_doc{num_documents}_ret{num_retrieved}_user{num_users}.pkl",
    )
    with open(load_path, "rb") as f:
        diff_inputs = pickle.load(f)

    difference_generation_prompt = (
        "You are a user behavior analyst. "
        "You are given the title and description of an item, along with the current user's review and other users' reviews of the same item. "
        "Your task is to identify stylistic and expressive differences between the current user's review and those of others, independent of content or opinion. "
        "Begin by evaluating the following predefined dimensions: "
        "[Writing Style] Definition: Lexical choices and syntactic structure. Difference Description: [Explain how the user's writing differs from others]. "
        "[Emotional Style] Definition: Affective tone (e.g., positive, neutral, negative). Difference Description: [Explain how the user's emotional tone differs]. "
        "[Semantic Style] Definition: Information density and contextual coherence. Difference Description: [Explain how the user's information organization differs]. "
        "Then, identify as many additional stylistic and expressive dimensions where the user's review noticeably diverges from those of others. "
        "Avoid referencing specific content or item-related opinions—focus solely on the review style. "
        "For each dimension, provide the following: "
        "1. Feature Name. 2. Definition: A concise explanation of the stylistic characteristic. 3. Difference Description: A clear description of how the user's review differs. "
        "Format your output strictly as follows: "
        "--- [Feature Name] Definition: ... Difference Description: ... --- (Repeat for all dimensions).\n"
    )
    diff_gen_pt = DeepSeekR1PromptTemplate()

    difference_validation_prompt = (
        "You are a user behavior analyst. "
        "You are given the title and description of an item, along with the current user's review and other users' reviews of the same item. "
        "Additionally, you are given a set of extracted difference descriptions spanning multiple feature dimensions that compare the current user's review with those of others. "
        "Your task is to filter these difference descriptions and retain only those that effectively distinguish stylistic and expressive differences between the current user's review and others. "
        "Format your output strictly as follows: "
        "--- [Feature Name] Definition: ... Difference Description: ... --- (Repeat for all valid dimensions).\n"
    )
    diff_valid_pt = DeepSeekR1PromptTemplate()

    # model_name = args.model_name
    model_name = get_local_model(args.model_name)

    # assert model_name.startswith("deepseek")
    sampling_params = SamplingParams(
        n=1,
        max_tokens=args.max_tokens,
        skip_special_tokens=True,
        temperature=args.temperature,
        top_p=0.95,
        seed=42,
    )
    llm = LLM(
        model_name,
        tensor_parallel_size=len(args.gpu_id.split(",")),
        gpu_memory_utilization=0.8,
        seed=42,
    )

    ## stage 1: difference_generation
    output_path = os.path.join(args.output_dir, args.method, category, "diff_outputs")
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(
        output_path,
        f"{category}_doc{num_documents}_ret{num_retrieved}_user{num_users}_{tag}.pkl",
    )
    diff_outputs = [[] for _ in range(len(main_dataset))]
    for i in range(num_retrieved):
        print(f"Stage 1, Iter {i}/{num_retrieved}\n")
        diff_input = [
            diff_gen_pt.build_prompt(difference_generation_prompt + diff_inp[i])
            for diff_inp in diff_inputs
        ]
        diff = llm.generate(diff_input, sampling_params)
        diff = [d.outputs[0].text.strip() for d in diff]
        for j, d in enumerate(diff):
            # output = d
            if "</think>" in d:
                output = d.split("</think>", 1)[1].strip()
            else:
                output = d
            diff_outputs[j].append(output)

    with open(save_path, "wb") as f:
        pickle.dump(diff_outputs, f)

    ## stage 2: difference_validation
    output_path = os.path.join(args.output_dir, args.method, category, "diff_valid_outputs")
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(
        output_path,
        f"{category}_doc{num_documents}_ret{num_retrieved}_user{num_users}_{tag}.pkl",
    )

    diff_valid_outputs = [[] for _ in range(len(main_dataset))]
    for i in range(num_retrieved):
        print(f"Stage 2, Iter {i}/{num_retrieved}\n")
        diff_input = [
            diff_valid_pt.build_prompt(
                difference_validation_prompt + diff_inp[i] + diff_out[i]
            )
            for diff_out, diff_inp in zip(diff_outputs, diff_inputs)
        ]
        diff = llm.generate(diff_input, sampling_params)
        diff = [d.outputs[0].text.strip() for d in diff]

        for j, d in enumerate(diff):
            # output = d
            if "</think>" in d:
                output = d.split("</think>", 1)[1].strip()
            else:
                output = d
            diff_valid_outputs[j].append(output)

    with open(save_path, "wb") as f:
        pickle.dump(diff_valid_outputs, f)

    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)
