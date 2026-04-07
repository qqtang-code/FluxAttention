import os
import datasets
import torch
import logging
import ast
import numpy as np
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass, field
from typing import Optional
import glob
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
import shutil

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CLASS_MAP = {
    "Single QA": 0,
    "MultiHop QA": 1,
    "Summarization": 2,
    "Code": 3,
    "In-Context Learning": 4,
}


@dataclass
class DataArguments:
    single_seq: bool = field(
        default=False, metadata={"help": "Override the length of the input"}
    )
    subsplit_length: Optional[int] = field(
        default=None, metadata={"help": "Split sequences into small lengths"}
    )
    per_device_varlen_padding: Optional[int] = field(
        default=4_294_967_296,
        metadata={"help": "Excess tokens for variable length attention"},
    )
    per_device_max_tokens: Optional[int] = field(
        default=4_294_967_296, metadata={"help": "Maximum number of tokens per device"}
    )
    apply_instruct_masks: bool = field(
        default=False,
        metadata={
            "help": "Whether to apply loss masks over the instructions (for instruction tuning). If enabled, will read the `mask` field in the data and set the corresponding labels to -100."
        },
    )


@dataclass
class PackedDataArguments:
    single_seq: bool = False
    subsplit_length: Optional[int] = None
    per_device_max_tokens: int = 128 * 1024
    apply_instruct_masks: bool = False
    prepack: bool = False
    streaming: bool = False
    min_seq_len: Optional[int] = 1000
    task_type: str = "pretrain"
    use_packing: bool = False
    data_cache_dir: Optional[str] = None
    preprocessing_num_workers: int = 32
    suffix: str = "qwen3_8b"


def _process_single_item(item, tokenizer, class_map, is_sft=False):

    ctx = item.get("context", "") or ""
    q = item.get("question", "") or ""
    a_text = item.get("answer", "") or ""

    if isinstance(ctx, str):
        ctx = ctx.replace("\ufeff", "")
    if isinstance(q, str):
        q = q.replace("\ufeff", "")
    if isinstance(a_text, str):
        a_text = a_text.replace("\ufeff", "")

    meta = item.get("metadata", {}) or {}
    task_type = "Other"
    is_prefix = True
    try:
        meta_dict = ast.literal_eval(meta) if isinstance(meta, str) else meta
        task_type = meta_dict.get("task", "Other")
        is_prefix = meta_dict.get("is_prefix", True)
    except Exception:
        pass

    separator = "\n\n"

    # Context (Segment ID 1)
    ctx_text = "\n" + ctx.rstrip()
    ctx_ids = tokenizer(ctx_text, add_special_tokens=False)["input_ids"]

    # Question (Segment ID 2)
    q_text = "\n" + q.lstrip()
    q_ids = tokenizer(q_text, add_special_tokens=False)["input_ids"]

    if is_prefix:
        user_text = q_text + "\n" + ctx_text

    else:
        user_text = ctx_text + "\n" + q_text

    if task_type == "Summarization":
        user_text = (
            "You are given several news passages. Write a one-page summary of all news."
            + user_text
            + "\n\nSummary:"
        )
    if task_type == "Code":
        user_text = "Please complete the code given below." + user_text

    messages = [{"role": "user", "content": user_text}]

    user_part_text = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False
    )
    user_part_ids = tokenizer(user_part_text, add_special_tokens=False)["input_ids"]
    user_len = len(user_part_ids)

    if a_text:
        messages.append({"role": "assistant", "content": a_text})

    full_text = tokenizer.apply_chat_template(
        messages, tokenize=False, enable_thinking=False
    )
    full_input_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]

    labels = list(full_input_ids)

    if is_sft:
        labels[:user_len] = [-100] * user_len
    else:
        pass

    if tokenizer.eos_token_id is not None and (
        not full_input_ids or full_input_ids[-1] != tokenizer.eos_token_id
    ):
        full_input_ids.append(tokenizer.eos_token_id)
        labels.append(tokenizer.eos_token_id)

    user_text_start = 0
    user_text_end = user_len - 1

    if a_text:
        a_start = user_len
        a_end = len(full_input_ids) - 1
    else:
        a_start = user_len
        a_end = len(full_input_ids) - 1

    if not is_sft and len(labels) > 0:
        labels[0] = -100

    range_ids = [
        0,
        0,
        user_text_start,
        user_text_end,
        user_text_start,
        user_text_end,
        a_start,
        a_end,
    ]
    class_id = class_map.get(task_type, 5)

    return {
        "input_ids": full_input_ids,
        "labels": labels,
        "task_id": class_id,
        "task_type": task_type,
        "range_ids": range_ids,
    }


def _finalize_pack(
    tokenizer, input_ids, labels, task_ids, lengths, task_types, range_ids
):
    seq_lengths = [0] + list(np.cumsum(lengths))

    return {
        "input_ids": input_ids,  # List[int]
        "labels": labels,  # List[int]
        "seq_lengths": seq_lengths,  # List[int]
        "task_ids": task_ids,  # List[int]
        "task_type": task_types,  # List[str]
        "range_ids": range_ids,  # List[int] [8]
    }


def worker_pack_chunk(
    chunk_dataset, tokenizer, max_seq_len, min_seq_len, worker_id, is_sft=False
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    local_packed_data = []

    # Buffers
    buf_input_ids = []
    buf_labels = []
    buf_task_ids = []
    buf_lengths = []
    buf_task_types = []
    buf_range_ids = []

    iterator = chunk_dataset
    if worker_id % 4 == 3:
        iterator = tqdm(
            chunk_dataset, desc=f"Worker {worker_id} Packing", position=worker_id
        )

    for item in iterator:
        processed = _process_single_item(item, tokenizer, CLASS_MAP, is_sft)

        p_input_ids = processed["input_ids"]
        p_len = len(p_input_ids)

        if p_len > max_seq_len or p_len < min_seq_len:
            continue

        if len(buf_input_ids) + p_len <= max_seq_len:
            buf_input_ids.extend(p_input_ids)
            buf_labels.extend(processed["labels"])
            buf_task_ids.append(processed["task_id"])
            buf_lengths.append(p_len)
            buf_task_types.append(processed["task_type"])
            buf_range_ids.append(processed["range_ids"])
        else:
            packed_item = _finalize_pack(
                tokenizer,
                buf_input_ids,
                buf_labels,
                buf_task_ids,
                buf_lengths,
                buf_task_types,
                buf_range_ids,
            )
            local_packed_data.append(packed_item)

            # Reset buffer
            buf_input_ids = list(p_input_ids)
            buf_labels = list(processed["labels"])
            buf_task_ids = [processed["task_id"]]
            buf_lengths = [p_len]
            buf_task_types = [processed["task_type"]]
            buf_range_ids = [processed["range_ids"]]

    if buf_input_ids:
        packed_item = _finalize_pack(
            tokenizer,
            buf_input_ids,
            buf_labels,
            buf_task_ids,
            buf_lengths,
            buf_task_types,
            buf_range_ids,
        )
        local_packed_data.append(packed_item)

    return local_packed_data


class PackedDataset(Dataset):
    def __init__(
        self,
        raw_dataset,
        tokenizer,
        max_seq_len=128 * 1024,
        min_seq_len=1000,
        cache_dir=None,
        num_proc=8,
        raw_path=None,
        suffix: str = None,
        is_sft: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.min_seq_len = min_seq_len
        self.packed_data = None

        self.cache_path = None
        # suffix = os.path.basename(tokenizer.name_or_path.rstrip("/"))

        suffix = suffix.lower()
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            cache_filename = f"{os.path.basename(raw_path)}_{suffix}_packed_maxseq{max_seq_len}.parquet"
            self.cache_path = os.path.join(cache_dir, cache_filename)

        if self.cache_path and os.path.exists(self.cache_path):
            try:
                self.packed_data = load_dataset(
                    "parquet",
                    data_files=self.cache_path,
                    split="train",
                )
                return
            except Exception as e:
                logger.warning(f"⚠️ error: ({e})...")
        packed_data_list = self._parallel_pack_dataset(raw_dataset, num_proc)

        keys = [
            "input_ids",
            "labels",
            "seq_lengths",
            "task_ids",
            "task_type",
            "range_ids",
        ]
        columnar = {k: [] for k in keys}
        for item in packed_data_list:
            for k in keys:
                columnar[k].append(item[k])
        self.packed_data = datasets.Dataset.from_dict(columnar)
        if self.cache_path:
            print(f"💾 save Parquet to: {self.cache_path} ...")
            try:
                self.packed_data.to_parquet(self.cache_path)
            except Exception as e:
                logger.error(f"❌ Error: {e}")

    def _parallel_pack_dataset(self, raw_dataset, num_proc):
        total_size = len(raw_dataset)
        num_proc = min(num_proc, total_size)
        if num_proc < 1:
            num_proc = 1

        print(f"Splitting dataset into {num_proc} chunks...")

        chunks = []
        for i in range(num_proc):
            chunks.append(
                raw_dataset.shard(num_shards=num_proc, index=i, contiguous=True)
            )

        futures = []
        with ProcessPoolExecutor(max_workers=num_proc) as executor:
            for i, chunk in enumerate(chunks):
                futures.append(
                    executor.submit(
                        worker_pack_chunk,
                        chunk,
                        self.tokenizer,
                        self.max_seq_len,
                        self.min_seq_len,
                        i,
                    )
                )

        results = []
        for f in tqdm(
            as_completed(futures), total=len(futures), desc="Waiting for workers"
        ):
            try:
                res = f.result()
                results.extend(res)
            except Exception as e:
                logger.error(f"Worker failed with error: {e}")
                raise e

        print(f"Packing done: Origin: {total_size} -> Packed: {len(results)}")
        return results

    def __len__(self):
        return len(self.packed_data)

    def __getitem__(self, idx):
        item = self.packed_data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"], dtype=torch.long),
            "labels": torch.tensor(item["labels"], dtype=torch.long),
            "seq_lengths": torch.tensor(item["seq_lengths"], dtype=torch.int32),
            "task_ids": torch.tensor(item["task_ids"], dtype=torch.long),
            "task_type": item["task_type"],
            "range_ids": torch.tensor(item["range_ids"], dtype=torch.long),
        }


def build_packed_dataset(paths: str, data_args, tokenizer=None, is_sft: bool = False):
    # if isinstance(paths, str):
    #     paths = [paths]

    parquet_files = []
    # for p in paths:
    if os.path.isdir(paths):
        parquet_files.extend(glob.glob(os.path.join(paths, "*.parquet")))
    elif os.path.isfile(paths):
        parquet_files.append(paths)

    print(f"******** {parquet_files} *******")
    if not parquet_files:
        raise ValueError("No parquet files found")

    # Load raw
    raw = load_dataset(
        "parquet",
        data_files=parquet_files,
        split="train",
        cache_dir=os.path.join(data_args.data_cache_dir, "raw")
        if data_args.data_cache_dir
        else None,
    )

    if "length" not in raw.column_names:
        print("Extracting 'length' from metadata for sorting...")

        raw = raw.map(
            lambda x: {
                "length": int(x["metadata"]["length"]) if x["metadata"]["length"] else 0
            },
            num_proc=data_args.preprocessing_num_workers,
            desc="Extracting lengths",
        )
    raw = raw.sort("length", reverse=False)

    max_len = data_args.per_device_max_tokens
    min_len = data_args.min_seq_len

    return PackedDataset(
        raw,
        tokenizer,
        max_seq_len=max_len,
        min_seq_len=min_len,
        cache_dir=data_args.data_cache_dir,
        num_proc=data_args.preprocessing_num_workers,
        raw_path=paths,
        suffix=data_args.suffix,
        is_sft=is_sft,
    )


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    path = "/workspace/mnt/qqt/public_data/length_subsets/32K-64K"
    data_args = PackedDataArguments(
        preprocessing_num_workers=32,
        data_cache_dir="/workspace/mnt/qqt/public_data/data_cache",
        per_device_max_tokens=65536,
        min_seq_len=1000,
        suffix="llama3.1-8b_64k_new",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "/workspace/mnt/hf_models/Llama-3.1-8B-Instruct", trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    import time

    print(f"\n⏱️  Start building dataset...")

    start_time = time.time()
    dataset = build_packed_dataset(
        paths=path,
        data_args=data_args,
        tokenizer=tokenizer,
        is_sft=False,
    )
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"⏱️  Done! Total time cost: {elapsed:.2f} s")

    print(f"\n✅ Dataset ready. Size: {len(dataset)}")
    item0 = dataset[0]
    print("\n--- Sample 0 Check ---")
    print(f"Keys: {item0.keys()}")
    print(f"Input IDs Shape: {item0['input_ids'].shape}")
    print(f"Task Types: {item0['task_type']}")
    print(f"Seq Lengths (cum): {item0['seq_lengths']}")
    print(f"Range ids: {item0['range_ids']}")
