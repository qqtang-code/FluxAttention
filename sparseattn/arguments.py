import argparse
import yaml
import ast
import os

import logging

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def parse_arguments():
    parser = argparse.ArgumentParser(description="evaluation on downstream tasks")
    parser.add_argument("--config", type=str, default=None, help="path to config file")
    parser.add_argument(
        "--tag", type=str, default="eval", help="tag to add to the output file"
    )

    # model setting
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="path to the model or model name",
    )
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument(
        "--use_vllm", action="store_true", help="whether to use vllm engine"
    )

    # data settings
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="comma separated list of dataset names",
    )
    parser.add_argument(
        "--demo_files",
        type=str,
        default=None,
        help="comma separated list of demo files",
    )
    parser.add_argument(
        "--test_files",
        type=str,
        default=None,
        help="comma separated list of test files",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None, help="path to save the predictions"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="whether to the saved file"
    )
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument(
        "--num_workers", type=int, default=4, help="number of workers for data loading"
    )

    # dataset specific settings
    parser.add_argument(
        "--popularity_threshold",
        type=int,
        default=3,
        help="popularity threshold for popqa, in log scale",
    )

    # evaluation settings
    parser.add_argument(
        "--shots", type=int, default=2, help="total number of ICL demos"
    )
    parser.add_argument(
        "--input_max_length",
        type=str,
        default="8192",
        help="the maximum number of tokens of the input, we truncate the end of the context; can be separated by comma to match the specified datasets",
    )

    # generation settings
    parser.add_argument(
        "--do_sample",
        type=ast.literal_eval,
        choices=[True, False],
        default=False,
        help="whether to use sampling (false is greedy), overwrites temperature",
    )
    parser.add_argument(
        "--generation_max_length",
        type=str,
        default="10",
        help="max number of tokens to generate, can be separated by comma to match the specified datasets",
    )
    parser.add_argument(
        "--generation_min_length",
        type=int,
        default=0,
        help="min number of tokens to generate",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.0, help="generation temperature"
    )
    parser.add_argument(
        "--top_p", type=float, default=1.0, help="top-p parameter for nucleus sampling"
    )
    parser.add_argument(
        "--stop_newline",
        type=ast.literal_eval,
        choices=[True, False],
        default=False,
        help="whether to stop generation at newline",
    )

    # model specific settings
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--no_cuda", action="store_true", help="disable cuda")
    parser.add_argument(
        "--no_bf16", action="store_true", help="disable bf16 and use fp32"
    )
    parser.add_argument(
        "--no_torch_compile", action="store_true", help="disable torchcompile"
    )
    parser.add_argument(
        "--use_chat_template",
        type=ast.literal_eval,
        choices=[True, False],
        default=False,
        help="whether to use chat template",
    )
    parser.add_argument(
        "--rope_theta", type=int, default=None, help="override rope theta"
    )

    # misc
    parser.add_argument("--debug", action="store_true", help="for debugging")
    parser.add_argument(
        "--count_tokens",
        action="store_true",
        help="instead of running generation, just count the number of tokens (only for HF models not API)",
    )

    # sparseattn
    parser.add_argument("--sparseattn", action="store_true", help="e2e eval")

    # duoattn
    parser.add_argument(
        "--duoattn", type=str, default=None, help="path to the duoattn pattern"
    )
    parser.add_argument(
        "--duoattn_sparsity",
        type=float,
        default=None,
        help="sparsity of the duoattn pattern",
    )
    parser.add_argument(
        "--duoattn_sink", type=int, default=128, help="sink size of the duoattn pattern"
    )
    parser.add_argument(
        "--duoattn_sliding",
        type=int,
        default=1024,
        help="sliding size of the duoattn pattern",
    )
    parser.add_argument(
        "--duoattn_chunk_prefilling",
        type=int,
        default=None,
        help="use chunk prefilling",
    )
    parser.add_argument(
        "--duoattn_flipping",
        action="store_true",
        help="whether to flip the duoattn pattern",
    )

    # fastprefill
    parser.add_argument(
        "--fastprefill_threshold",
        type=float,
        default=0.95,
        help="threshold for fastprefill",
    )
    parser.add_argument(
        "--fastprefill_print_detail",
        type=bool,
        default=False,
        help="whether to print detail for fastprefill",
    )
    parser.add_argument(
        "--fastprefill_stride", type=int, default=16, help="stride for fastprefill"
    )
    parser.add_argument(
        "--fastprefill_metric", type=str, default=None, help="metric for fastprefill"
    )

    # minference, snapkv and pyramidkv
    parser.add_argument(
        "--minference",
        type=str,
        default=None,
        help="which method to use with minference",
    )
    parser.add_argument(
        "--minference_model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B-Instruct",
        help="the model name to use with minference",
    )
    parser.add_argument(
        "--minference_chunk_prefilling",
        type=int,
        default=None,
        help="use chunk prefilling",
    )
    parser.add_argument(
        "--minference_sparsity",
        type=float,
        default=None,
        help="token sparsity - overrides max_capacity_prompt",
    )
    parser.add_argument(
        "--minference_window_size",
        type=int,
        default=32,
        help="window size for minference",
    )
    parser.add_argument(
        "--minference_max_capacity_prompt",
        type=int,
        default=4096,
        help="max capacity prompt for minference",
    )
    parser.add_argument(
        "--minference_chunking_patch",
        action="store_true",
        help="patch in the last `k` tokens for all chunk",
    )
    parser.add_argument(
        "--minference_grouped_eviction",
        action="store_true",
        help="grouped eviction for pyramid/snapkv",
    )
    parser.add_argument(
        "--minference_compress_group_kvs",
        action="store_true",
        help="compress group kvs for pyramid/snapkv",
    )

    # locret
    parser.add_argument(
        "--locret_bin_file", type=str, default=None, help="path to the locret bin file"
    )
    parser.add_argument(
        "--locret_sparsity", type=float, default=None, help="KV sparsity for locret"
    )
    parser.add_argument(
        "--locret_budget_size",
        type=int,
        default=None,
        help="budget size for locret; overriden by locret_sparsity",
    )
    parser.add_argument(
        "--locret_local_len", type=int, default=100, help="local length for locret"
    )
    parser.add_argument(
        "--locret_stabilizers", type=int, default=2500, help="stabilizers for locret"
    )
    parser.add_argument(
        "--locret_chunk_prefilling",
        type=int,
        default=None,
        help="chunk size for locret",
    )

    args = parser.parse_args()
    config = yaml.safe_load(open(args.config)) if args.config is not None else {}
    parser.set_defaults(**config)
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = f"output/{os.path.basename(args.model_name_or_path)}"

    if args.rope_theta is not None:
        args.output_dir = args.output_dir + f"-override-rope{args.rope_theta}"

    if not args.do_sample and args.temperature != 0.0:
        args.temperature = 0.0
        logger.info("overwriting temperature to 0.0 since do_sample is False")

    return args
