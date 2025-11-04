import os
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import re
import json
import argparse
import pickle
import math
from pathlib import Path
from typing import List, Dict, Any, Optional
from collections import defaultdict, Counter


KEEP_THRESH = 2      
CONSIST_K   = 3     

ALPHA_BY_CAT = {
    "semantics": 0.16,
    "emotion":   0.15,
    "structure": 0.14,
    "pragmatics":0.13,
    "writing":   0.12,
}

CAT_W = {"semantics":1.0, "structure":1.0, "writing":1.0, "pragmatics":1.0, "emotion":1.0}

COVERAGE_BONUS = 0.1

WRITING_FREQ_CAPS = {
    "writing/sentence-complexity": 2,
    "writing/exclamations": 1,
    "writing/vocabulary": 2,
}

SRC_GAIN_STEP = 0.05
SRC_GAIN_MAX  = 1.15

RELIABILITY_MIN = 0.85

ALLOWED_CATS = ["Writing", "Emotion", "Semantics", "Structure", "Pragmatics"]

CONTROLLED_DUPKEYS: Dict[str, List[str]] = {
    "Writing": [
        "writing/sentence-complexity/higher",
        "writing/sentence-complexity/lower",
        "writing/vocabulary/richer",
        "writing/vocabulary/simpler",
        "writing/exclamations/more",
        "writing/exclamations/fewer",
        "writing/conversational-tone/more",
    ],
    "Emotion": [
        "emotion/tone/positive/more",
        "emotion/tone/neutral/more",
        "emotion/tone/negative/more",
        "emotion/intensity/higher",
        "emotion/intensity/lower",
        "emotion/politeness/hedging/more",
    ],
    "Semantics": [
        "semantics/info-density/higher",
        "semantics/info-density/lower",
        "semantics/analytical-reasoning/more",
        "semantics/narrative-episodic/more",
        "semantics/specificity/more",
        "semantics/specificity/less",
        "semantics/causal-reasoning/more",
        "semantics/thematic-depth/more",
        "semantics/comparative-contrast/more",
        "semantics/synthesis-integration/more",
    ],
    "Structure": [
        "structure/paragraphs/more",
        "structure/paragraphs/fewer",
        "structure/transitions/clearer",
        "structure/headings-lists/more",
        "structure/no-headings-lists/true",
        "structure/argumentation-clearer",
        "structure/sectioning/clearer",
    ],
    "Pragmatics": [
        "pragmatics/quotes/more",
        "pragmatics/rhetorical-questions/more",
        "pragmatics/disclosure/more",
        "pragmatics/reader-address/more",
        "pragmatics/imperatives/more",
    ],
}


Qwen2PromptTemplate = None
get_local_model = None

try:
    from utils.templates import Qwen2PromptTemplate  
except Exception:
    try:
        from utils import templates as _tpl  
        Qwen2PromptTemplate = getattr(_tpl, "Qwen2PromptTemplate", None)
    except Exception:
        Qwen2PromptTemplate = None

try:
    from utils.get_local_model import get_local_model  
except Exception:
    try:
        from utils import get_local_model  
    except Exception:
        get_local_model = None

if Qwen2PromptTemplate is None:
    class Qwen2PromptTemplate:
        def __init__(self, system_prompt: str = ""):
            self.system_prompt = system_prompt
        def build_prompt(self, user_content: str) -> str:
            return (
                "<|im_start|>system\n" + self.system_prompt.strip() + "\n<|im_end|>\n"
                "<|im_start|>user\n" + user_content.strip() + "\n<|im_end|>\n"
                "<|im_start|>assistant\n"
            )

from vllm import LLM, SamplingParams
from transformers import set_seed


def _controlled_key_table() -> str:
    lines = []
    for cat in ALLOWED_CATS:
        keys = CONTROLLED_DUPKEYS[cat]
        lines.append(f"- {cat}:")
        for k in keys:
            lines.append(f"  * {k}")
    return "\n".join(lines)

ALIASES_HINT = """
CANONICALIZATION HINTS (map synonyms to the exact dup_key):
- Longer/complex/compound sentences, more clauses → writing/sentence-complexity/higher
- Shorter/simpler sentences → writing/sentence-complexity/lower
- Richer vocabulary / varied lexicon → writing/vocabulary/richer
- Simpler vocabulary → writing/vocabulary/simpler
- (More|Fewer) exclamation marks/“!” → writing/exclamations/(more|fewer)
- Conversational/casual/chatty tone → writing/conversational-tone/more

- More positive/cheerful/optimistic tone → emotion/tone/positive/more
- More neutral/objective/less emotional → emotion/tone/neutral/more
- More negative/critical tone → emotion/tone/negative/more
- Stronger emotion/excited/passionate → emotion/intensity/higher
- Calmer/less intense → emotion/intensity/lower
- More hedging/politeness (“I think”, “maybe”) → emotion/politeness/hedging/more

- Denser information/more details/context → semantics/info-density/higher
- Fewer details/brief/concise → semantics/info-density/lower
- More analytical/reasoning/comparison/causality → semantics/analytical-reasoning/more
- More narrative/storytelling/episodic → semantics/narrative-episodic/more
- More specific references/names/numbers → semantics/specificity/more
- More general/abstract → semantics/specificity/less

- More causal links (“because/therefore/so/leads to”) → semantics/causal-reasoning/more
- Deeper theme/motif analysis (“themes, motifs, symbolism”) → semantics/thematic-depth/more
- More compare/contrast structure (“while/whereas/versus”) → semantics/comparative-contrast/more
- Better synthesis/integration across evidence → semantics/synthesis-integration/more

- More paragraphs/clear sections → structure/paragraphs/more
- Fewer paragraphs → structure/paragraphs/fewer
- Clearer transitions/connectives (“however”, “therefore”) → structure/transitions/clearer
- Headings or bullet/numbered lists → structure/headings-lists/more
- No headings/lists at all → structure/no-headings-lists/true
- Clearer argumentation/claims-evidence → structure/argumentation-clearer
- Clearer sectioning/subsections → structure/sectioning/clearer

- More direct quotes/citations → pragmatics/quotes/more
- More rhetorical questions (“?” used to persuade) → pragmatics/rhetorical-questions/more
- Disclosure/acknowledgement (ARC, thanks publisher) → pragmatics/disclosure/more
- Addressing reader (“you”, “dear reader”) → pragmatics/reader-address/more
- More imperatives (“Read this”, “Go try”) → pragmatics/imperatives/more
""".strip()

FEWSHOT = r"""
EXAMPLE A (mapping):
RAW: "Compared with others, the user writes in longer, more complex sentences, and uses more exclamation marks!"
JSON cells (abbrev):
[
  {"feature_name":"Longer Sentences","difference":"The user uses longer, more complex sentences than others.","category":"Writing","is_valid":true,"invalid_reason":"","dup_key":"writing/sentence-complexity/higher","evidence_span":"longer, more complex sentences"},
  {"feature_name":"More Exclamations","difference":"The user uses more exclamation marks than others.","category":"Writing","is_valid":true,"invalid_reason":"","dup_key":"writing/exclamations/more","evidence_span":"more exclamation marks"}
]

EXAMPLE B (invalid):
RAW: "The user likes this author more than others."
Reason: content preference (entity/opinion). -> is_valid=false, invalid_reason="content_bias".
""".strip()

SYSTEM_PROMPT = (
    "You are an expert reviewer of writing styles. You receive noisy text describing how a TARGET user's review "
    "DIFFERS from OTHER users on the same item. Extract VALID, ATOMIC difference units and normalize them into a "
    "controlled taxonomy.\n"
    "\n"
    "CATEGORIES (expression-only): Writing / Emotion / Semantics / Structure / Pragmatics. "
    "Exclude content preference or entity liking (invalid: content_bias). "
    "A VALID unit MUST be comparative (target vs others: more/less/higher/lower/fewer).\n"
    "Atomicity: ONE property per unit. If one sentence has two properties, split into two units.\n"
    "Inside this cell, DEDUP identical dup_key (keep only one). If a base attribute appears with opposite directions, "
    "keep the side with stronger evidence within this cell.\n"
    "\n"
    "CONTROLLED TAXONOMY (dup_key MUST be chosen EXACTLY from the list below; DO NOT invent new keys):\n"
    f"{_controlled_key_table()}\n"
    "\n"
    f"{ALIASES_HINT}\n"
    "\n"
    f"{FEWSHOT}\n"
    "\n"
    "OUTPUT STRICT JSON with fields:\n"
    "{\n"
    '  \"file\": \"<file_name>\", \"j\": <int>, \"i\": <int>,\n'
    '  \"cells\": [\n'
    "    {\n"
    '      \"feature_name\": \"<short specific phrase>\",\n'
    '      \"difference\": \"<concise comparative sentence (target vs others)>\",\n'
    '      \"category\": \"Writing|Emotion|Semantics|Structure|Pragmatics\",\n'
    '      \"is_valid\": true|false,\n'
    '      \"invalid_reason\": \"\" | \"content_bias\" | \"no_comparison\" | \"unclear\" | \"format_error\",\n'
    '      \"dup_key\": \"<one key from CONTROLLED TAXONOMY>\",\n'
    '      \"evidence_span\": \"<short snippet from the text>\"\n'
    "    }\n"
    "  ],\n"
    '  \"cell_summary\": {\n'
    '    \"valid_count_dedup\": <int>,\n'
    '    \"by_category\": {\"Writing\": <int>, \"Emotion\": <int>, \"Semantics\": <int>, \"Structure\": <int>, \"Pragmatics\": <int>}\n'
    "  }\n"
    "}\n"
    "Rules:\n"
    "- Only output JSON, no extra text. If nothing valid, return empty cells[] with zeros in the summary.\n"
    "- category MUST be one of the five categories; dup_key MUST be chosen from the CONTROLLED TAXONOMY for that category.\n"
    "- Within this cell, do not repeat the same dup_key; output at most 12 units.\n"
)

USER_PROMPT_TEMPLATE = (
    "FILE: {file_name}\n"
    "CELL: j={j}, i={i}\n"
    "RAW TEXT:\n"
    "{raw_text}\n"
    "\n"
    "Now produce the STRICT JSON as specified."
)

def build_prompt(file_name: str, j: int, i: int, raw_text: str) -> str:
    pt = Qwen2PromptTemplate(system_prompt=SYSTEM_PROMPT)
    user = USER_PROMPT_TEMPLATE.format(file_name=file_name, j=j, i=i, raw_text=raw_text)
    return pt.build_prompt(user)


FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)```", re.IGNORECASE)

def _strip_noise(text: str) -> str:
    return (
        (text or "")
        .replace("</think>", "")
        .replace("<|im_start|>", "")
        .replace("<|im_end|>", "")
        .strip()
    )

def _parse_first_json_anywhere(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    text = _strip_noise(text)
    for m in FENCE_RE.finditer(text):
        cand = (m.group(1) or "").strip()
        if not cand:
            continue
        try:
            return json.loads(cand)
        except Exception:
            continue
    s = text.find("{")
    if s == -1:
        return None
    closes = [m.start() for m in re.finditer(r"\}", text[s:])]
    for idx in reversed(closes):
        frag = text[s : s + idx + 1].strip()
        if len(frag) < 2 or len(frag) > 200_000:
            continue
        try:
            return json.loads(frag)
        except Exception:
            continue
    return None

def _norm_category(raw: Any) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    s = s.split("/", 1)[0].strip().title()
    return s if s in ALLOWED_CATS else None

def _ensure_summary(obj: Dict[str, Any]) -> Dict[str, Any]:
    """只做格式修复（尊重 LLM 判定），保证 cell_summary 存在。"""
    cs = obj.get("cell_summary")
    byc: Dict[str, int] = {c: 0 for c in ALLOWED_CATS}
    vcd: int = 0

    if isinstance(cs, dict) and "by_category" in cs:
        try:
            for c in ALLOWED_CATS:
                byc[c] = int(cs["by_category"].get(c, 0) or 0)
            vcd = int(cs.get("valid_count_dedup", 0) or 0)
        except Exception:
            pass

    if vcd == 0:
        seen = set()
        for cell in obj.get("cells", []) or []:
            if not isinstance(cell, dict):
                continue
            if not cell.get("is_valid", False):
                continue
            cat = _norm_category(cell.get("category"))
            if not cat:
                continue
            byc[cat] += 1
            dup = str(cell.get("dup_key") or "").strip().lower()
            if dup:
                seen.add(dup)
        vcd = len(seen)

    obj["cell_summary"] = {
        "valid_count_dedup": int(vcd or 0),
        "by_category": {c: int(byc.get(c, 0) or 0) for c in ALLOWED_CATS},
    }
    return obj


def base_attr(key: str) -> str:
    parts = (key or "").lower().split("/")
    return "/".join(parts[:2]) if len(parts) >= 2 else (key or "").lower()

def dir_sign(key: str) -> int:
    k = (key or "").lower()
    if any(t in k for t in ("/higher", "/more", "/true")):
        return +1
    if any(t in k for t in ("/lower", "/less", "/fewer")):
        return -1
    return 0

def _log_gain(f: int, alpha: float) -> float:
    if f <= 1:
        return 1.0
    return 1.0 + float(alpha) * math.log1p(f - 1)

def aggregate_rows(results: List[Dict[str, Any]], N: int) -> Dict[str, Any]:
    row_key_freq   = [Counter() for _ in range(N)]                 
    row_key_cat    = [{} for _ in range(N)]                        
    row_key_srcs   = [defaultdict(set) for _ in range(N)]          
    row_attr_vote  = [defaultdict(int) for _ in range(N)]          
    row_attr_srcs  = [defaultdict(lambda: {"pos": set(), "neg": set()}) for _ in range(N)] 

    kept_cells = 0
    dropped_by_conflict = 0

    for obj in results:
        j = int(obj.get("j", 0))
        i = int(obj.get("i", 0))
        if j < 0 or j >= N:
            continue
        col_seen = set()
        for cell in obj.get("cells", []) or []:
            if not cell.get("is_valid", False):
                continue
            cat = _norm_category(cell.get("category"))
            if not cat or cat not in ALLOWED_CATS:
                continue
            dup = (cell.get("dup_key") or "").strip().lower()
            if not dup:
                continue
            if dup not in CONTROLLED_DUPKEYS[cat]:
                continue
            if dup in col_seen:
                continue
            col_seen.add(dup)

            kept_cells += 1
            row_key_freq[j][dup] += 1
            row_key_cat[j][dup] = cat.lower()
            row_key_srcs[j][dup].add(i)

            b = base_attr(dup)
            s = dir_sign(dup)
            row_attr_vote[j][b] += s
            if s > 0:
                row_attr_srcs[j][b]["pos"].add(i)
            elif s < 0:
                row_attr_srcs[j][b]["neg"].add(i)

    attr_rows_total = Counter()
    attr_rows_conflict = Counter()
    for j in range(N):
        for b, pn in row_attr_srcs[j].items():
            pos_n = len(pn["pos"])
            neg_n = len(pn["neg"])
            if pos_n + neg_n > 0:
                attr_rows_total[b] += 1
                if pos_n > 0 and neg_n > 0:
                    attr_rows_conflict[b] += 1
    attr_reliability = {}
    for b, tot in attr_rows_total.items():
        conf = attr_rows_conflict.get(b, 0)
        r = 1.0 - (conf / max(1, tot))
        attr_reliability[b] = max(RELIABILITY_MIN, r)  

    row_unique_valid_sum = 0
    row_unique_k2_sum    = 0
    row_unique_by_cat_sum = {c.lower(): 0 for c in CAT_W}
    row_freq_score_sum   = 0.0

    row_unique_valid_list = []
    row_unique_k2_list    = []
    row_freq_score_list   = []

    for j in range(N):
        valid_dir = {}  
        for b, v in row_attr_vote[j].items():
            pos_n = len(row_attr_srcs[j][b]["pos"])
            neg_n = len(row_attr_srcs[j][b]["neg"])

            if abs(v) >= KEEP_THRESH:
                valid_dir[b] = "pos" if v > 0 else "neg"
            else:
                if pos_n >= 2 and neg_n <= 1:
                    valid_dir[b] = "pos"
                elif neg_n >= 2 and pos_n <= 1:
                    valid_dir[b] = "neg"
                else:
                    valid_dir[b] = None

        all_attrs = set(map(base_attr, row_key_freq[j].keys()))
        kept_attrs = {b for b, d in valid_dir.items() if d is not None}
        dropped_by_conflict += len(all_attrs - kept_attrs)

        hit_cat = set()
        row_u = 0
        row_k2 = 0
        row_score = 0.0

        for k, f in row_key_freq[j].items():
            b = base_attr(k)
            d = valid_dir.get(b)
            if d is None:
                continue
            s = dir_sign(k)
            if (d == "pos" and s <= 0) or (d == "neg" and s >= 0):
                continue

            cat = row_key_cat[j][k]              
            alpha = ALPHA_BY_CAT.get(cat, 0.14)
            f_eff = f
            if cat == "writing" and b in WRITING_FREQ_CAPS:
                f_eff = min(f, WRITING_FREQ_CAPS[b])

            gain = _log_gain(f_eff, alpha)

            src_cnt = len(row_key_srcs[j][k])
            src_gain = min(SRC_GAIN_MAX, 1.0 + SRC_GAIN_STEP * max(0, src_cnt - 1))

            rel = attr_reliability.get(b, 1.0)

            contrib = CAT_W.get(cat, 1.0) * gain * src_gain * rel
            row_score += contrib

            row_u += 1
            if f >= CONSIST_K:
                row_k2 += 1
            row_unique_by_cat_sum[cat] += 1

            if COVERAGE_BONUS > 0 and cat not in hit_cat:
                row_score += COVERAGE_BONUS
                hit_cat.add(cat)

        row_unique_valid_sum += row_u
        row_unique_k2_sum    += row_k2
        row_freq_score_sum   += row_score

        row_unique_valid_list.append(row_u)
        row_unique_k2_list.append(row_k2)
        row_freq_score_list.append(row_score)

    kept = kept_cells
    dropped = dropped_by_conflict
    conflict_drop_rate = float(dropped) / float(max(1, kept + dropped))

    return {
        "row_unique_valid_sum": int(row_unique_valid_sum),
        "row_unique_k2_sum": int(row_unique_k2_sum),
        "row_unique_by_category_sum": {k: int(v) for k, v in row_unique_by_cat_sum.items()},
        "row_freq_score_sum": round(row_freq_score_sum, 4),
        "row_unique_valid_mean": round(sum(row_unique_valid_list) / max(1, N), 4),
        "row_unique_k2_mean": round(sum(row_unique_k2_list) / max(1, N), 4),
        "row_freq_score_mean": round(sum(row_freq_score_list) / max(1, N), 4),
        "conflict_drop_rate": round(conflict_drop_rate, 4),
        "KEEP_THRESH": KEEP_THRESH,
        "CONSIST_K": CONSIST_K,
        "ALPHA_BY_CAT": ALPHA_BY_CAT,
        "CAT_W": CAT_W,
        "COVERAGE_BONUS": COVERAGE_BONUS,
        "WRITING_FREQ_CAPS": WRITING_FREQ_CAPS,
        "SRC_GAIN_STEP": SRC_GAIN_STEP,
        "SRC_GAIN_MAX": SRC_GAIN_MAX,
        "RELIABILITY_MIN": RELIABILITY_MIN,
    }


def _parse_gpus(g: str):
    toks = [t for t in re.split(r"[,\s]+", (g or "").strip()) if t != ""]
    return ",".join(toks), len(toks)

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", required=True,
        help="Path to a diff_valid_outputs PKL, e.g. /home/.../Books_doc0_ret8_user4_DeepSeek-R1-Distill-Qwen-32B.pkl")
    ap.add_argument("--out_dir", required=True,
        help="Output dir for llm_analysis.jsonl and summary.json")
    ap.add_argument("--model_name", default="Qwen/Qwen2.5-32B-Instruct",
        help="Judge LLM (Qwen instruct). Resolved by utils.get_local_model if available.")
    ap.add_argument("--gpus", default="0,1,2,3",
        help="CUDA_VISIBLE_DEVICES list, e.g. 0,1,2,3")
    ap.add_argument("--tp", type=int, default=None, help="tensor_parallel_size; default=len(gpus)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=12, help="vLLM generate() batch size")
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--gpu_mem_util", type=float, default=0.7, help="vLLM gpu_memory_utilization")
    ap.add_argument("--num_samples", type=int, default=-1, help="Limit rows (j). -1 means all.")
    ap.add_argument("--num_retrieved", type=int, default=-1, help="Limit cols (i). -1 means all.")
    return ap.parse_args()

def main():
    args = parse_args()

    visible, num_gpus = _parse_gpus(args.gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    tp = args.tp or num_gpus or 1
    set_seed(args.seed)
    print(f"[INFO] GPUs={visible}  -> tensor_parallel_size={tp}")

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    analysis_path = out_dir / "llm_analysis.jsonl"
    summary_path  = out_dir / "summary.json"

    with open(args.file, "rb") as f:
        matrix = pickle.load(f)
    if not isinstance(matrix, list):
        raise TypeError("PKL payload must be List[List[str]]")
    N = len(matrix)

    model_id = args.model_name
    if get_local_model is not None:
        try:
            model_id = get_local_model(args.model_name)
        except Exception:
            model_id = args.model_name

    llm = LLM(
        model=model_id,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_mem_util,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        stop=["<|im_end|>"],
    )

    file_name = os.path.basename(args.file)

    jobs: List[Dict[str, Any]] = []
    max_j = args.num_samples if args.num_samples and args.num_samples > 0 else N
    max_i = args.num_retrieved if args.num_retrieved and args.num_retrieved > 0 else None

    for j in range(min(max_j, N)):
        row = matrix[j]
        if not isinstance(row, list):
            raise TypeError(f"Row {j} is not a list.")
        Rj = len(row)
        Ri = min(max_i, Rj) if max_i else Rj
        for i in range(Ri):
            raw = row[i]
            raw = "" if raw is None else str(raw)
            prompt = build_prompt(file_name, j, i, raw)
            jobs.append({"prompt": prompt, "meta": {"file": file_name, "j": j, "i": i}})

    print(f"[INFO] Will process subset: {len(jobs)} cells (num_samples={max_j}, num_retrieved={max_i or 'all'})")

    results: List[Dict[str, Any]] = []
    for k in range(0, len(jobs), args.batch):
        batch_prompts = [x["prompt"] for x in jobs[k : k + args.batch]]
        outs = llm.generate(batch_prompts, sampling)
        for out, job in zip(outs, jobs[k : k + args.batch]):
            text = (out.outputs[0].text or "").strip()
            meta = job["meta"]

            obj = _parse_first_json_anywhere(text)
            if obj is None:
                obj = {
                    "file": meta["file"],
                    "j": meta["j"],
                    "i": meta["i"],
                    "cells": [],
                    "cell_summary": {
                        "valid_count_dedup": 0,
                        "by_category": {c: 0 for c in ALLOWED_CATS},
                    },
                    "raw_llm_text": text,
                }
            else:
                obj["file"] = meta["file"]
                obj["j"] = obj.get("j", meta["j"])
                obj["i"] = obj.get("i", meta["i"])
                _ensure_summary(obj)

            results.append(obj)

    with open(analysis_path, "w", encoding="utf-8") as fw:
        for obj in results:
            fw.write(json.dumps(obj, ensure_ascii=False) + "\n")

    total_valid = sum(int((obj.get("cell_summary", {}) or {}).get("valid_count_dedup", 0) or 0) for obj in results)
    cat_counts = {c: 0 for c in ALLOWED_CATS}
    for obj in results:
        byc = (obj.get("cell_summary", {}) or {}).get("by_category", {}) or {}
        for c in ALLOWED_CATS:
            cat_counts[c] += int(byc.get(c, 0) or 0)
    total_cats = sum(cat_counts.values()) or 1
    proportions = {c: cat_counts[c] / total_cats for c in ALLOWED_CATS}

    agg = aggregate_rows(results, N)

    summary = {
        "file": file_name,
        "N": N,
        "R": len(matrix[0]) if N > 0 and isinstance(matrix[0], list) else 0,
        "model": model_id,
        "gpus": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "tp": getattr(getattr(llm, "llm_engine", None), "model_executor", None)
        and getattr(llm.llm_engine.model_executor, "tensor_parallel_size", None),


        "total_valid_features_legacy": int(total_valid),
        "category_counts_legacy": {c: int(cat_counts[c]) for c in ALLOWED_CATS},
        "category_proportions_legacy": proportions,

        "aggregation": agg,
        "controlled_taxonomy": CONTROLLED_DUPKEYS, 
        "params": {
            "KEEP_THRESH": KEEP_THRESH,
            "CONSIST_K": CONSIST_K,
            "ALPHA_BY_CAT": ALPHA_BY_CAT,
            "CAT_W": CAT_W,
            "COVERAGE_BONUS": COVERAGE_BONUS,
            "WRITING_FREQ_CAPS": WRITING_FREQ_CAPS,
            "SRC_GAIN_STEP": SRC_GAIN_STEP,
            "SRC_GAIN_MAX": SRC_GAIN_MAX,
            "RELIABILITY_MIN": RELIABILITY_MIN,
        },
    }

    with open(summary_path, "w", encoding="utf-8") as fw:
        json.dump(summary, fw, ensure_ascii=False, indent=2)

    print(f"[OK] LLM per-cell results -> {analysis_path}")
    print(f"[OK] File-level summary  -> {summary_path}")
    print("[OK] Key new metrics:",
          "row_unique_k2_sum=", agg["row_unique_k2_sum"],
          "row_freq_score_sum=", agg["row_freq_score_sum"],
          "conflict_drop_rate=", agg["conflict_drop_rate"])

if __name__ == "__main__":
    main()
