import os

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
import sys
import argparse
import pickle
import warnings
from collections import defaultdict

from tqdm import tqdm
import torch.distributed as dist
from datasets import load_from_disk
from transformers import set_seed
from utils.templates import Qwen2PromptTemplate, DeepSeekR1PromptTemplate
from vllm import SamplingParams, LLM

from utils.utils import postprocess_output
from utils.templates import Qwen2PromptTemplate
from utils.preprocess import create_prompt_generator, GeneralDataset

from utils.get_local_model import get_local_model

warnings.filterwarnings("ignore")
set_seed(42)
if dist.is_initialized():
    dist.barrier()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", default="0,2")

    parser.add_argument("--method", default="DRP")

    parser.add_argument("--dataset", choices=["val", "test"], default="test")

    parser.add_argument(
        "--diff_model_name",
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
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
    )

    parser.add_argument("--model_name", default="Qwen/Qwen2.5-14B-Instruct")
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
    diff_model_name = args.diff_model_name

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
    prompt_generator = create_prompt_generator(
        num_documents=num_documents,
        num_retrieved=num_retrieved,
        num_users=num_users,
        asin_reviewers_map=asin_reviewers_map,
    )
    dataset = GeneralDataset(main_dataset, user_profile_map, asin_map, prompt_generator)
    dataset = [
        (gene_creator, summ_creator, out)
        for gene_creator, summ_creator, out in tqdm(
            dataset, desc="Data-processing", total=len(dataset)
        )
    ]
    gene_creators, summ_creators, references = zip(*dataset)

    load_name = f"{category}_doc{num_documents}_ret{num_retrieved}_user{num_users}_{diff_model_name.split('/')[1]}.pkl"
    load_path = os.path.join(
        args.output_dir, args.method, category, "diff_outputs", load_name
    )


    with open(load_path, "rb") as f:
        diff_outputs = pickle.load(f)


    model_name = get_local_model(args.model_name)
    
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
        gpu_memory_utilization=0.9,
        seed=42,
    )

    item_intro = "an item" if num_retrieved == 1 else f"{num_retrieved} items"
    summarizer_system_prompt = (
        f"Given titles and descriptions of {item_intro}, along with the differences between the current user's review and other users' reviews for each item, and the current user's past reviews, "
        f"generate a profile summary of the current user.\n"
        f"The summary should be formatted as follows:\n"
        f"[Summary]: <summary>"
    )
    summarizer_pt = Qwen2PromptTemplate(system_prompt=summarizer_system_prompt)
    summ_inputs = [
        summ_creator(diff) for diff, summ_creator in zip(diff_outputs, summ_creators)
    ]
    summ_inputs = [summarizer_pt.build_prompt(summ_inp) for summ_inp in summ_inputs]
    print("summary stage")
    summaries = llm.generate(summ_inputs, sampling_params)
    summaries = [s.outputs[0].text.strip() for s in summaries]

    output_path = os.path.join(args.output_dir, args.method, category, "summ_outputs")
    os.makedirs(output_path, exist_ok=True)
    save_path = os.path.join(
        output_path,
        f"{category}_doc{num_documents}_ret{num_retrieved}_user{num_users}_{diff_model_name.split('/')[1]}.pkl",
    )

    with open(save_path, "wb") as f:
        pickle.dump(summaries, f)

    generator_system_prompt = (
        f"Given the title and description of an item, along with the current user's past reviews and profile summary, and the output review rating and review title, "
        f"generate a personalized item review for the current user.\n"
        f"The review should be formatted as follows:\n"
        f"[Review]: <review>"
    )
    generator_pt = Qwen2PromptTemplate(system_prompt=generator_system_prompt)

    inputs = [
        gene_creator(summ) for summ, gene_creator in zip(summaries, gene_creators)
    ]
    inputs = [generator_pt.build_prompt(inp) for inp in inputs]

    print("generation stage")
    predictions = llm.generate(inputs, sampling_params)
    predictions = [
        postprocess_output(prediction.outputs[0].text)
        for prediction in tqdm(
            predictions, desc="Post-processing", total=len(predictions)
        )
    ]

    output_path = os.path.join(args.output_dir, args.method, category, "final_outputs")
    os.makedirs(output_path, exist_ok=True)
    save_name = f"{category}_doc{num_documents}_ret{num_retrieved}_user{num_users}_{diff_model_name.split('/')[1]}.txt"

    save_path = os.path.join(output_path, save_name)
    with open(save_path, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(pred + "\n---------------------------------\n")

    if dist.is_initialized():
        dist.destroy_process_group()
    sys.exit(0)
