import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import argparse
import pickle
import warnings

import torch.distributed as dist
from datasets import load_from_disk
from transformers import set_seed
from utils.templates import Qwen2PromptTemplate
from vllm import SamplingParams, LLM
from utils.get_local_model import get_local_model


warnings.filterwarnings("ignore")
set_seed(42)
if dist.is_initialized():
    dist.barrier()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", default="0,1,2,3")

    parser.add_argument("--method", default="DRP")
    parser.add_argument("--dataset", choices=["val", "test"], default="test")

    parser.add_argument(
        "--model_name",
        default="Qwen/Qwen2.5-14B-Instruct",
        choices=[
            "Qwen/Qwen2.5-1.5B-Instruct",
            "Qwen/Qwen2.5-7B-Instruct",
            "Qwen/Qwen2.5-14B-Instruct",
            "Qwen/Qwen2.5-32B-Instruct",
        ],
    )
    parser.add_argument("--max_tokens", type=int, default=4096)
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
        default="Qwen2.5-1.5B-Instruct",
        choices=[
            "Qwen2.5-1.5B-Instruct",
            "Qwen2.5-7B-Instruct",
            "Qwen2.5-14B-Instruct",
            "Qwen2.5-32B-Instruct",
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
    diff_gen_pt = Qwen2PromptTemplate(system_prompt=difference_generation_prompt)

    difference_validation_prompt = (
        "You are a user behavior analyst. "
        "You are given the title and description of an item, along with the current user's review and other users' reviews of the same item. "
        "Additionally, you are given a set of extracted difference descriptions spanning multiple feature dimensions that compare the current user's review with those of others. "
        "Your task is to filter these difference descriptions and retain only those that effectively distinguish stylistic and expressive differences between the current user's review and others. "
        "Format your output strictly as follows: "
        "--- [Feature Name] Definition: ... Difference Description: ... --- (Repeat for all valid dimensions).\n"
    )
    diff_valid_pt = Qwen2PromptTemplate(system_prompt=difference_validation_prompt)

    # model_name = args.model_name
    model_name = get_local_model(args.model_name)
    # base_name = args.model_name.split("/")[0]
    # model_tag = args.model_name.split("/")[-1]
    # assert model_name.startswith("Qwen")
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
        diff_input = [diff_gen_pt.build_prompt(diff_inp[i]) for diff_inp in diff_inputs]
        diff = llm.generate(diff_input, sampling_params)
        diff = [d.outputs[0].text.strip() for d in diff]
        for j, d in enumerate(diff):
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
            diff_valid_pt.build_prompt(user_message=diff_inp[i] + diff_out[i])
            for diff_out, diff_inp in zip(diff_outputs, diff_inputs)
        ]
        diff = llm.generate(diff_input, sampling_params)
        diff = [d.outputs[0].text.strip() for d in diff]

        for j, d in enumerate(diff):
            output = d
            diff_valid_outputs[j].append(output)

    with open(save_path, "wb") as f:
        pickle.dump(diff_valid_outputs, f)

    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)
