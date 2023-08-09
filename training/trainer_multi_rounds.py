#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : trainer.py
@Author  : wenge-research
'''

import os, sys
os.environ["TOKENIZERS_PARALLELISM"] = "false"
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import logging
import click
import numpy as np
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from datetime import datetime
from datasets import Dataset, load_dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)
from .consts import (
    DEFAULT_INPUT_MODEL,
    DEFAULT_SEED,
    END_KEY,
    INSTRUCTION_KEY,
    RESPONSE_KEY,
    INTRO_KEY,
    INTRO
)

logger = logging.getLogger(__name__)
ROOT_PATH = Path(__file__).parent.parent

class DataCollatorForCompletionOnlyLM_Multi_Rounds(DataCollatorForLanguageModeling):
    def torch_call(self, examples: List[Union[List[int], Any, Dict[str, Any]]]) -> Dict[str, Any]:
        batch = super().torch_call(examples)

        # The prompt ends with the response key plus a newline.  We encode this and then try to find it in the
        # sequence of tokens.  This should just be a single token.
        yayi_token_ids = self.tokenizer(RESPONSE_KEY)["input_ids"][0]
        human_token_ids = self.tokenizer(INSTRUCTION_KEY)["input_ids"][0]

        labels = batch["labels"].clone()

        for i in range(len(examples)):

            response_start, response_end = None, None
            yayi_position = np.where(batch["labels"][i] == yayi_token_ids)[0].tolist()
            human_position = np.where(batch["labels"][i] == human_token_ids)[0].tolist()
            labels[i, :human_position[0]+3] = -100
            for response_start,response_end in zip(human_position,yayi_position):
                if response_start is None or response_end is None:
                    raise RuntimeError(
                        f'Could not find response key {yayi_token_ids}/{human_token_ids} in token IDs'
                    )
                labels[i, response_start:response_end+3] = -100

        batch["labels"] = labels
        return batch

def preprocess_batch(batch: Dict[str, List], tokenizer: AutoTokenizer, max_length: int) -> dict:
    return tokenizer(
        batch["text"],
        max_length=max_length,
        truncation=True,
    )


def load_training_dataset__multi_rounds(path_or_dataset: str = "data/yayi_train_example_multi_rounds.json") -> Dataset:
    logger.info(f"Loading dataset from {path_or_dataset}")
    dataset = load_dataset("json", data_files=path_or_dataset)["train"]
    logger.info(dataset)
    logger.info("Found %d rows", dataset.num_rows)

    def _add_text(rec):
        content_format = "\n{}\n\n"
        conversations = rec.get("conversations")
        system = rec.get("system",INTRO) 
        system = system if system!="" else INTRO
        dialogue_list = [INTRO_KEY+":",content_format.format(system)]

        for conversation in conversations:
            if conversation["from"] == "human":
                dialogue_list.append(INSTRUCTION_KEY+":")
                dialogue_list.append(content_format.format(conversation["value"]))
            elif conversation["from"] == "yayi":
                dialogue_list.append(RESPONSE_KEY+":")
                dialogue_list.append(content_format.format(conversation["value"])+END_KEY)

        rec["text"] = "".join(dialogue_list)
        return rec

    dataset = dataset.map(_add_text)
    return dataset


def load_tokenizer(pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL) -> PreTrainedTokenizer:
    logger.info(f"Loading tokenizer for {pretrained_model_name_or_path}")
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [INTRO_KEY, INSTRUCTION_KEY, RESPONSE_KEY, END_KEY]})
    return tokenizer


def load_model(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> AutoModelForCausalLM:
    logger.info(f"Loading model for {pretrained_model_name_or_path}")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path, trust_remote_code=True, use_cache=False if gradient_checkpointing else True
    )
    return model


def get_model_tokenizer(
    pretrained_model_name_or_path: str = DEFAULT_INPUT_MODEL, *, gradient_checkpointing: bool = False
) -> Tuple[AutoModelForCausalLM, PreTrainedTokenizer]:
    tokenizer = load_tokenizer(pretrained_model_name_or_path)
    model = load_model(pretrained_model_name_or_path, gradient_checkpointing=gradient_checkpointing)
    model.resize_token_embeddings(len(tokenizer))

    return model, tokenizer


def preprocess_dataset(tokenizer: AutoTokenizer, max_length: int, seed=DEFAULT_SEED, path_or_dataset=None) -> Dataset:
    """Loads the training dataset and tokenizes it so it is ready for training.

    Args:
        tokenizer (AutoTokenizer): Tokenizer tied to the model.
        max_length (int): Maximum number of tokens to emit from tokenizer.

    Returns:
        Dataset: HuggingFace dataset
    """

    dataset = load_training_dataset__multi_rounds(path_or_dataset=path_or_dataset)

    logger.info("Preprocessing dataset")
    _preprocessing_function = partial(preprocess_batch, max_length=max_length, tokenizer=tokenizer)
    dataset = dataset.map(
        _preprocessing_function,
        batched=True,
        remove_columns=["conversations", "system", "text"],
    )
    logger.info(f"datasets after processing: {dataset}")

    # Make sure we don't have any truncated records, as this would mean the end keyword is missing.
    logger.info("Processed dataset has %d rows", dataset.num_rows)
    dataset = dataset.filter(lambda rec: len(rec["input_ids"]) < max_length)
    logger.info("Processed dataset has %d rows after filtering for truncated records", dataset.num_rows)

    logger.info("Shuffling dataset")
    dataset = dataset.shuffle(seed=seed)

    logger.info("Done preprocessing")

    return dataset


def train(
    *,
    data_path: str,
    input_model: str,
    local_output_dir: str,
    dbfs_output_dir: str,
    epochs: int,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    lr: float,
    seed: int,
    deepspeed: str,
    gradient_checkpointing: bool,
    local_rank: str,
    bf16: bool,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    test_size: Union[float, int],
    save_total_limit: int,
    warmup_steps: int,
):
    set_seed(seed)

    # Create dir for saving logs and checkpoints
    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M")
    model_name = "YAYI_CHAT"
    checkpoint_dir_name = f"{model_name}_{timestamp}"
    local_output_dir = os.path.join(local_output_dir, checkpoint_dir_name)
    os.makedirs(local_output_dir, exist_ok=True)
    model, tokenizer = get_model_tokenizer(
        pretrained_model_name_or_path=input_model, gradient_checkpointing=gradient_checkpointing
    )

    # Use the same max length that the model supports.
    max_length = None
    for length_setting in ["n_positions", "max_position_embeddings", "seq_length"]:
        max_length = getattr(model.config, length_setting, None)
        if max_length:
            logger.info(f"Found max lenth: {max_length}")
            break
    if not max_length:
        max_length = 1024
        logger.info(f"Using default max length: {max_length}")

    # Data processing
    hf_data_dir = data_path.replace(".json","")
    if os.path.exists(hf_data_dir):
        logger.info("Load dataset from cache.")
        split_dataset = load_from_disk(hf_data_dir)
    else:
        logger.info("Load dataset from disk.")
        processed_dataset = preprocess_dataset(tokenizer=tokenizer, max_length=max_length, seed=seed, path_or_dataset=data_path)
        split_dataset = processed_dataset.train_test_split(test_size=test_size, seed=seed)

    logger.info("Train data size: %d", split_dataset["train"].num_rows)
    logger.info("Test data size: %d", split_dataset["test"].num_rows)

    data_collator = DataCollatorForCompletionOnlyLM_Multi_Rounds(
        tokenizer=tokenizer, mlm=False, return_tensors="pt", pad_to_multiple_of=8
    )

    training_args = TrainingArguments(
        output_dir=local_output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        fp16=False,
        bf16=bf16,
        learning_rate=lr,
        num_train_epochs=epochs,
        deepspeed=deepspeed,
        gradient_checkpointing=gradient_checkpointing,
        logging_dir=f"{local_output_dir}/runs",
        logging_strategy="steps",
        logging_steps=logging_steps,
        evaluation_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=save_total_limit,
        load_best_model_at_end=False,
        report_to="tensorboard",
        disable_tqdm=False,
        remove_unused_columns=False,
        local_rank=local_rank,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine"
    )

    logger.info("Instantiating Trainer")

    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=split_dataset["train"],
        eval_dataset=split_dataset["test"],
        data_collator=data_collator,
    )

    logger.info("Training")
    trainer.train()

    logger.info(f"Saving Model to {local_output_dir}")
    trainer.save_model(output_dir=local_output_dir)

    if dbfs_output_dir:
        logger.info(f"Saving Model to {dbfs_output_dir}")
        trainer.save_model(output_dir=dbfs_output_dir)

    logger.info("Done.")


@click.command()
@click.option("--data-path", type=str, help="Input data to fine tune", default="data/yayi_train_example_multi_rounds.json")
@click.option("--input-model", type=str, help="Input model to fine tune", default=DEFAULT_INPUT_MODEL)
@click.option("--local-output-dir", type=str, help="Write directly to this local path", required=True)
@click.option("--dbfs-output-dir", type=str, help="Sync data to this path on DBFS")
@click.option("--epochs", type=int, default=3, help="Number of epochs to train for.")
@click.option("--per-device-train-batch-size", type=int, default=1, help="Batch size to use for training.")
@click.option("--per-device-eval-batch-size", type=int, default=1, help="Batch size to use for evaluation.")
@click.option("--test-size", type=int, default=1, help="Number of test records for evaluation, or ratio of test records.")
@click.option("--warmup-steps", type=int, default=1, help="Number of steps to warm up to learning rate")
@click.option("--logging-steps", type=int, default=10, help="How often to log")
@click.option("--eval-steps", type=int, default=50, help="How often to run evaluation on test records")
@click.option("--save-steps", type=int, default=100, help="How often to checkpoint the model")
@click.option("--save-total-limit", type=int, default=10, help="Maximum number of checkpoints to keep on disk")
@click.option("--lr", type=float, default=1e-5, help="Learning rate to use for training.")
@click.option("--seed", type=int, default=DEFAULT_SEED, help="Seed to use for training.")
@click.option("--deepspeed", type=str, default=None, help="Path to deepspeed config file.")
@click.option(
    "--gradient-checkpointing/--no-gradient-checkpointing",
    is_flag=True,
    default=True,
    help="Use gradient checkpointing?",
)
@click.option(
    "--local_rank",
    type=str,
    default=True,
    help="Provided by deepspeed to identify which instance this process is when performing multi-GPU training.",
)
@click.option("--bf16", type=bool, default=True, help="Whether to use bf16 (preferred on A100's).")
def main(**kwargs):
    train(**kwargs)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S"
    )
    try:
        main()
    except Exception:
        logger.exception("main failed")
        raise