import os

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
import re
import argparse
from collections import defaultdict, Counter
from datasets import load_dataset, load_from_disk
from datasets import Dataset, DatasetDict


def run_from_scratch():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--category",
        type=str,
        choices=["CDs_and_Vinyl", "Movies_and_TV", "Books"],
        default="CDs_and_Vinyl",
    )
    parser.add_argument("--download", type=bool, default=True)

    parser.add_argument("--data_dir", type=str, default="./data", help="Root directory to save datasets")
    parser.add_argument("--cache_dir", type=str, default="./.cache/huggingface/datasets", help="Local HuggingFace cache directory")

    args = parser.parse_args()
    category = args.category
    download = args.download

    if download:
        print(f"Downloading datasets for category: {category}")
        review_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_review_{category}",
            split="full",
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        )
        review_dataset.save_to_disk(
            f"{args.data_dir}/raw_review_{category}"
        )

        meta_dataset = load_dataset(
            "McAuley-Lab/Amazon-Reviews-2023",
            f"raw_meta_{category}",
            split="full",
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        )
        
        meta_dataset.save_to_disk(
            f"{args.data_dir}/raw_meta_{category}"
        )

    else:
        print(f"Loading datasets for category: {category}")
        review_dataset = load_from_disk(
            f"{args.data_dir}/raw_review_{category}"
        )
        meta_dataset = load_from_disk(
            f"{args.data_dir}/raw_meta_{category}"
        )


    df = review_dataset.to_pandas()[["user_id", "asin", "timestamp", "text"]]
    user_counts = df["user_id"].value_counts()
    print(user_counts.head(10))
    print(user_counts.describe())

    def review_filter(x):
        return (
            x["text"]
            and len(x["text"]) >= 200
            and x["rating"]
            and x["title"]
            and x["timestamp"]
        )

    def create_description(x):
        desc = x["description"]
        if not desc or (desc[0] == "Product Description" and len(desc) == 1):
            return ""
        if desc[0] == "Product Description":
            return " ".join(desc[1:])
        return " ".join(desc)

    def meta_filter(x):
        return (
            100 <= len(create_description(x)) <= 2000
            and x["title"]
            and x["categories"]
            and x["rating_number"]
        )

    meta_columns = [
        col for col in meta_dataset.column_names if col not in {"title", "description"}
    ]
    meta_dataset = meta_dataset.filter(
        meta_filter, num_proc=64, load_from_cache_file=False
    )
    meta_dataset = meta_dataset.map(
        lambda x: {
            "asin": x["parent_asin"],
            "title": x["title"],
            "description": create_description(x),
        },
        num_proc=64,
        remove_columns=meta_columns,
        load_from_cache_file=False,
    )

    valid_asins = set(meta_dataset["asin"])

    review_columns = [
        col
        for col in review_dataset.column_names
        if col not in {"user_id", "asin", "title", "text", "rating", "timestamp"}
    ]
    review_dataset = review_dataset.filter(
        review_filter, num_proc=64, load_from_cache_file=False
    )
    review_dataset = review_dataset.map(
        lambda x: {
            "user_id": x["user_id"],
            "asin": x["parent_asin"],
            "title": x["title"],
            "text": x["text"],
            "rating": x["rating"],
            "timestamp": x["timestamp"],
        },
        num_proc=64,
        remove_columns=review_columns,
        load_from_cache_file=False,
    )
    review_dataset = review_dataset.filter(
        lambda x: x["asin"] in valid_asins, num_proc=64, load_from_cache_file=False
    )

    def filter_users_and_items(dataset):
        cnt = 0
        while True:
            cnt += 1
            user_review = defaultdict(list)
            asin_user = defaultdict(set)

            def collect(x):
                user_review[x["user_id"]].append(x)
                asin_user[x["asin"]].add(x["user_id"])
                return x

            dataset = dataset.map(collect, load_from_cache_file=False)

            for user, reviews in user_review.items():
                reviews.sort(key=lambda r: r["timestamp"], reverse=True)
                seen_asin, seen_text = set(), set()
                deduped = []
                for r in reviews:
                    if r["asin"] not in seen_asin and r["text"] not in seen_text:
                        seen_asin.add(r["asin"])
                        seen_text.add(r["text"])
                        deduped.append(r)
                user_review[user] = deduped

            asin_user_count = Counter(
                {asin: len(users) for asin, users in asin_user.items()}
            )
            user_review_count = Counter(
                {user: len(reviews) for user, reviews in user_review.items()}
            )

            asin_user = {
                asin: users
                for asin, users in asin_user.items()
                if asin_user_count[asin] >= 8
            }
            user_review = {
                user: reviews
                for user, reviews in user_review.items()
                if 18 <= user_review_count[user] <= 500
            }

            new_dataset = dataset.filter(
                lambda x: x["asin"] in asin_user and x["user_id"] in user_review,
                load_from_cache_file=False,
            )

            print(len(new_dataset), len(dataset))
            if len(new_dataset) == len(dataset):
                print(f"Finished in {cnt} iterations")
                break
            dataset = new_dataset

        return dataset, user_review

    review_dataset, user_review = filter_users_and_items(review_dataset)


    def clean_and_split_reviews():
        asin_user = defaultdict(set)
        for user_id, reviews in user_review.items():
            reviews.sort(key=lambda x: x["timestamp"], reverse=False)
            for r in reviews:
                r["text"] = re.sub(r"\s+", " ", r["text"].strip())

        user_review_split = {}
        asin_user_new = defaultdict(set)
        for user_id, reviews in user_review.items():
            if len(reviews) <= 10:
                continue
            profile, input_ = reviews[:-10], reviews[-10:]
            for r in profile:
                asin_user_new[r["asin"]].add(user_id)
            user_review_split[user_id] = (profile, input_)

        cnt = 0
        while True:
            cnt += 1
            asin_user_count = Counter(
                {asin: len(users) for asin, users in asin_user_new.items()}
            )
            asin_user_new = {
                asin: users
                for asin, users in asin_user_new.items()
                if asin_user_count[asin] >= 5
            }

            new_user_review_split = {}
            for user_id, (profile, input_) in user_review_split.items():
                filtered_profile = [r for r in profile if r["asin"] in asin_user_new]
                if len(filtered_profile) >= 8:
                    new_user_review_split[user_id] = (filtered_profile, input_)
            if len(new_user_review_split) == len(user_review_split):
                print(f"Finished in {cnt} iterations")
                break

            user_review_split = new_user_review_split
            asin_user_new = defaultdict(set)
            for user_id, (profile, _) in user_review_split.items():
                for r in profile:
                    asin_user_new[r["asin"]].add(user_id)

        return user_review_split

    user_review_split = clean_and_split_reviews()

    train_data, val_data, test_data = [], [], []
    for user_id, (profile, inputs) in user_review_split.items():
        for i in range(len(inputs) - 2):
            train_data.append((user_id, profile + inputs[:i], inputs[i]))
        val_data.append((user_id, profile + inputs[:-2], inputs[-2]))
        test_data.append((user_id, profile + inputs[:-1], inputs[-1]))

    def convert_to_dataset(data):
        user_ids, profiles, targets = zip(*data)
        return Dataset.from_dict(
            {
                "user_id": list(user_ids),
                "profile": list(profiles),
                "data": list(targets),
            }
        )

    train_dataset = convert_to_dataset(train_data)
    val_dataset = convert_to_dataset(val_data)
    test_dataset = convert_to_dataset(test_data)

    main_dataset = DatasetDict(
        {
            "train": train_dataset,
            "val": val_dataset,
            "test": test_dataset,
        }
    )
    meta_dataset = DatasetDict({"full": meta_dataset})

    main_dataset.save_to_disk(f"{args.data_dir}/review_{category}")
    meta_dataset.save_to_disk(f"{args.data_dir}/meta_{category}")


def run_direct_download():
    parser = argparse.ArgumentParser()
    parser.add_argument("--category", type=str, choices=["CDs_and_Vinyl", "Movies_and_TV", "Books"], default="Books")
    parser.add_argument("--download", type=bool, default=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--cache_dir", type=str, default="./.cache/huggingface/datasets")
    args = parser.parse_args()

    category = args.category
    download = args.download

    split = args.split
    
    if download:
        print(f"Downloading datasets for category: {category}")
        main_dataset = load_dataset(
            "SnowCharmQ/DPL-main",
            category,
            split=split,
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        )
        main_dataset.save_to_disk(f"{args.data_dir}/DRP-main/{category}/{split}")
        

        meta_dataset = load_dataset(
            "SnowCharmQ/DPL-meta",
            category,
            split="full",
            trust_remote_code=True,
            cache_dir=args.cache_dir,
        )

        meta_dataset.save_to_disk(f"{args.data_dir}/DRP-meta/{category}/full")


    else:
        print(f"Loading datasets for category: {category}")
        
        main_dataset = load_from_disk(
            f"{args.data_dir}/DRP-main/{category}/{split}"
        )
        meta_dataset = load_from_disk(
            f"{args.data_dir}/DRP-meta/{category}/full"
        )



if __name__ == "__main__":
    # run_from_scratch()
    run_direct_download()
