import logging
import os
import sys
import torch
import datasets
import transformers
import functools

from transformers import (
    AutoConfig,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
)

import torch
from transformers import LlamaForCausalLM, AutoTokenizer

from .modeling_flash_llama import PawLlamaForCausalLM, PawLlamaConfig, LlamaModel
from .modeling_flash_qwen import (
    PawQwen3ForCausalLM,
    PawQwen3Config,
    Qwen3Model,
)

from .lh_trainer import Trainer
# from .lh_trainer_nsa import Trainer as NSATrainer


from .dataset_packing import PackedDataArguments as DataArguments
from .dataset_packing import build_packed_dataset
from .dataset_packing import logger as dataset_logger
from .script_arguments import ScriptArguments, TrainingArguments


from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy
import torch.distributed as dist

from transformers.trainer_utils import get_last_checkpoint
import json

from csv import reader

import multiprocessing

# from fla.models.nsa import AutoModelForCausalLM as NSAAutoModelForCausalLM

logger = logging.getLogger(__name__)


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of script_args, for a cleaner separation of concerns.
    parser = HfArgumentParser((ScriptArguments, TrainingArguments, DataArguments))
    script_args, training_args, data_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    dataset_logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"Additional arguments {script_args}")
    logger.info(f"Data arguments {data_args}")
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)
    tokenizer = AutoTokenizer.from_pretrained(
        script_args.tokenizer_name or script_args.model_name_or_path,
        cache_dir=script_args.cache_dir,
        use_fast=script_args.use_fast_tokenizer,
        revision=script_args.model_revision,
        use_auth_token=True if script_args.use_auth_token else None,
        enable_thinking=True if script_args.use_thinking else False,
    )

    # Determine model type and load appropriate config
    if "qwen" in script_args.model_name_or_path.lower():
        config = PawQwen3Config.from_pretrained(
            script_args.config_name or script_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            revision=script_args.model_revision,
            use_auth_token=True if script_args.use_auth_token else None,
            toggle_type=training_args.toggle_type,
            retrieval_mode=training_args.retrieval_mode,
            local_window_size=training_args.context_window_if_toggled,
            sink_size=training_args.sink_size,
            disable_linear_regularization_term=training_args.disable_linear_regularization_term,
            pooling_mode=training_args.pooling_mode,
            use_task_emb_for_mask=training_args.use_task_emb_for_mask,
            enable_lambda_task=training_args.enable_lambda_task,
            enable_ada_sparsity=training_args.enable_ada_sparsity,
            use_softmax=training_args.use_softmax,
            pool_size=training_args.pool_size,
        )
    elif "llama" in script_args.model_name_or_path.lower():
        config = PawLlamaConfig.from_pretrained(
            script_args.config_name or script_args.model_name_or_path,
            cache_dir=script_args.cache_dir,
            revision=script_args.model_revision,
            use_auth_token=True if script_args.use_auth_token else None,
            toggle_type=training_args.toggle_type,
            retrieval_mode=training_args.retrieval_mode,
            local_window_size=training_args.context_window_if_toggled,
            sink_size=training_args.sink_size,
            topk_k=training_args.topk_k,
            disable_linear_regularization_term=training_args.disable_linear_regularization_term,
            pooling_mode=training_args.pooling_mode,
            use_task_emb_for_mask=training_args.use_task_emb_for_mask,
            enable_lambda_task=training_args.enable_lambda_task,
            enable_ada_sparsity=training_args.enable_ada_sparsity,
            use_softmax=training_args.use_softmax,
            pool_size=training_args.pool_size,
        )
    else:
        raise ValueError(
            f"Model name {script_args.model_name_or_path} does not contain. "
            "Please provide a valid model name."
        )
    if script_args.config_overrides:
        logger.info(f"Overriding config: {script_args.config_overrides}")
        config.update_from_string(script_args.config_overrides)
        logger.info(f"New config: {config}")

    if script_args.config_overrides_json:
        logger.info(f"Overriding config: {script_args.config_overrides_json}")
        config.update(json.loads(script_args.config_overrides_json))
        logger.info(f"New config: {config}")

    config.pad_token_id = 0

    if script_args.model_name_or_path:
        # Determine model type and load appropriate model
        if (
            training_args.attention_type is not None
            and "nsa" in training_args.attention_type
        ):
            model = LlamaForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                from_tf=bool(".ckpt" in script_args.model_name_or_path),
                cache_dir=script_args.cache_dir,
                revision=script_args.model_revision,
                use_auth_token=True if script_args.use_auth_token else None,
                torch_dtype=torch.bfloat16,
            )
        elif "qwen" in script_args.model_name_or_path.lower():
            model = PawQwen3ForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                from_tf=bool(".ckpt" in script_args.model_name_or_path),
                config=config,
                cache_dir=script_args.cache_dir,
                revision=script_args.model_revision,
                use_auth_token=True if script_args.use_auth_token else None,
            )
        elif "llama" in script_args.model_name_or_path.lower():
            model = PawLlamaForCausalLM.from_pretrained(
                script_args.model_name_or_path,
                from_tf=bool(".ckpt" in script_args.model_name_or_path),
                config=config,
                cache_dir=script_args.cache_dir,
                revision=script_args.model_revision,
                use_auth_token=True if script_args.use_auth_token else None,
            )
        else:
            raise ValueError(
                f"Model name {script_args.model_name_or_path} does not contain. "
                "Please provide a valid model name."
            )
    else:
        logger.warning(f"Initializing new PawLlamaForCausalLM from scratch")
        # Determine model type and initialize appropriate model
        if "qwen" in script_args.model_name_or_path.lower():
            model = PawQwen3ForCausalLM(config)
        elif "llama" in script_args.model_name_or_path.lower():
            model = PawLlamaForCausalLM(config)
        else:
            raise ValueError(
                f"Model name {script_args.model_name_or_path} does not contain. "
                "Please provide a valid model name."
            )

    def init_all_routers(module):
        if module.__class__.__name__ in ["LlamaModel", "Qwen3Model"]:
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

        for child in module.children():
            init_all_routers(child)

    init_all_routers(model)

    logger.info(f"Model: {model}")

    assert training_args.max_steps is not None, "max_steps must be set!"

    # load_datasets
    if training_args.do_train:
        train_dataset = build_packed_dataset(
            script_args.tokenized_mds_train[0],
            tokenizer=tokenizer,
            data_args=data_args,
            is_sft=False,
        )

        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        sp_size = training_args.seq_parallel_size

        dp_size = world_size // sp_size
        dp_rank = global_rank // sp_size

        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset=train_dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=True,
            seed=training_args.seed,
            drop_last=True,
        )

        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=None,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
            drop_last=True,
        )

    if training_args.do_eval:
        eval_dataset = build_dataset(
            script_args.tokenized_mds_validation[0],
            tokenizer=tokenizer,
            data_args=data_args,
        )
        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        sp_size = training_args.seq_parallel_size

        dp_size = world_size // sp_size
        dp_rank = global_rank // sp_size

        from torch.utils.data.distributed import DistributedSampler

        sampler = DistributedSampler(
            dataset=eval_dataset,
            num_replicas=dp_size,
            rank=dp_rank,
            shuffle=False,
            seed=training_args.seed,
        )

        eval_dataloader = torch.utils.data.DataLoader(
            dataset=eval_dataset,
            batch_size=1,
            sampler=sampler,
            collate_fn=None,
            num_workers=training_args.dataloader_num_workers,
            pin_memory=training_args.dataloader_pin_memory,
        )

    # Initialize our Trainer
    if (
        training_args.attention_type is not None
        and "nsa" in training_args.attention_type
    ):
        # trainer = NSATrainer(
        #     model=model,
        #     args=training_args,
        #     train_dataset=train_dataset if training_args.do_train else None,
        #     eval_dataset=eval_dataset if training_args.do_eval else None,
        #     tokenizer=tokenizer,
        #     data_collator=data_collator,
        #     log_loss=script_args.should_log_loss,
        # )
        pass
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset if training_args.do_train else None,
            eval_dataset=eval_dataset if training_args.do_eval else None,
            tokenizer=tokenizer,
            # data_collator=data_collator,
            log_loss=script_args.should_log_loss,
        )
    if training_args.do_train:
        trainer.train_dataloader = train_dataloader
        logger.info(
            "Successfully injected CustomDistributedStratifiedSampler into Trainer."
        )

    if training_args.do_eval:
        trainer.eval_dataloader = eval_dataloader
        logger.info(
            "Successfully injected CustomDistributedStratifiedSampler into Trainer."
        )

    if trainer.is_fsdp_enabled:
        # Identify which modules have "_fsdp_wrap" attribute set to True and wrap these
        def fsdp_policy_fn(module):
            return getattr(module, "_fsdp_wrap", False)

        auto_wrap_policy = functools.partial(
            lambda_auto_wrap_policy, lambda_fn=fsdp_policy_fn
        )
        trainer.accelerator.state.fsdp_plugin.auto_wrap_policy = auto_wrap_policy

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()

        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()


if __name__ == "__main__":
    main()
