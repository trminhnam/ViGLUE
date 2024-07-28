import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from typing import Dict

import torch
import transformers
from peft import PeftConfig, PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    AutoConfig,
    AutoModelForSeq2SeqLM,
    TextGenerationPipeline,
    Text2TextGenerationPipeline,
    pipeline,
)

COMPUTE_DTYPE_MAPPING = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


# Source: https://github.com/bofenghuang/stanford_alpaca/blob/eb5b171d9b103a12a8e14e0edca9cbc45fe1d512/train.py#L75-L95
def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.
    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def has_attr_and_true(obj, attr):
    return hasattr(obj, attr) and getattr(obj, attr)


def get_attr_or_default(obj, attr, default=None):
    return getattr(obj, attr) if hasattr(obj, attr) else default


def load_model_with_peft_and_tokenizer(model_args):
    ### QUANTIZATION CONFIG ###
    bnb_config = None
    if has_attr_and_true(model_args, "load_in_8bit") or has_attr_and_true(
        model_args, "load_in_4bit"
    ):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=get_attr_or_default(model_args, "load_in_8bit", False),
            load_in_4bit=get_attr_or_default(model_args, "load_in_4bit", False),
            bnb_4bit_quant_type=get_attr_or_default(
                model_args, "bnb_4bit_quant_type", "fp4"
            ),
            bnb_4bit_compute_dtype=COMPUTE_DTYPE_MAPPING[
                get_attr_or_default(model_args, "bnb_4bit_compute_dtype", "float32")
            ],
            bnb_4bit_use_double_quant=get_attr_or_default(
                model_args, "bnb_4bit_use_double_quant", False
            ),
        )
        print(f"Quantization config: {bnb_config}")

    ### Load PEFT Parameters ###
    if model_args.peft_name_or_path is not None:
        peft_config = PeftConfig.from_pretrained(
            model_args.peft_name_or_path,
            subfolder=model_args.peft_name_or_path_subfolder,
            cache_dir=model_args.cache_dir,
        )

        model_config = AutoConfig.from_pretrained(
            peft_config.base_model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        # from model config to model class
        architecture = model_config.architectures[0].lower()
        if any(x in architecture for x in ["gpt", "causal"]):
            model_class = AutoModelForCausalLM
        elif any(x in architecture for x in ["encoder-decoder", "seq2seq"]):
            model_class = AutoModelForSeq2SeqLM
        else:
            raise ValueError(f"Unsupported architecture {architecture} for PEFT model.")

        model = model_class.from_pretrained(
            peft_config.base_model_name_or_path,
            quantization_config=bnb_config,
            cache_dir=model_args.cache_dir,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            model_args.peft_name_or_path,
            subfolder=model_args.peft_name_or_path_subfolder,
            is_trainable=False,
            device_map="auto",
            cache_dir=model_args.cache_dir,
        )
    else:
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=model_args.cache_dir,
        )
        # from model config to model class
        architecture = model_config.architectures[0].lower()
        if any(x in architecture for x in ["gpt", "causal"]):
            model_class = AutoModelForCausalLM
        elif any(
            x in architecture
            for x in [
                "encoder-decoder",
                "seq2seq",
                "t5",
                "bart",
                "conditionalgeneration",
            ]
        ):
            model_class = AutoModelForSeq2SeqLM
        else:
            raise ValueError(f"Unsupported architecture {architecture}")

        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            subfolder=model_args.model_name_or_path_subfolder,
            quantization_config=bnb_config,
            cache_dir=model_args.cache_dir,
            device_map="auto",
        )
    model.eval()

    ### Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.peft_name_or_path or model_args.model_name_or_path,
        padding_side="left",
        use_fast=get_attr_or_default(model_args, "use_fast_tokenizer", False),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def load_pipeline(model_args):
    ### QUANTIZATION CONFIG ###
    bnb_config = None
    if has_attr_and_true(model_args, "load_in_8bit") or has_attr_and_true(
        model_args, "load_in_4bit"
    ):
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=get_attr_or_default(model_args, "load_in_8bit", False),
            load_in_4bit=get_attr_or_default(model_args, "load_in_4bit", False),
            bnb_4bit_quant_type=get_attr_or_default(
                model_args, "bnb_4bit_quant_type", "fp4"
            ),
            bnb_4bit_compute_dtype=COMPUTE_DTYPE_MAPPING[
                get_attr_or_default(model_args, "bnb_4bit_compute_dtype", "float32")
            ],
            bnb_4bit_use_double_quant=get_attr_or_default(
                model_args, "bnb_4bit_use_double_quant", False
            ),
        )
        print(f"Quantization config: {bnb_config}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.peft_name_or_path or model_args.model_name_or_path,
        trust_remote_code=True,
        padding_side="left",
    )

    if model_args.peft_name_or_path is not None:
        peft_config = PeftConfig.from_pretrained(
            model_args.peft_name_or_path,
            subfolder=model_args.peft_name_or_path_subfolder,
            trust_remote_code=True,
            cache_dir=model_args.cache_dir,
        )

        model_config = AutoConfig.from_pretrained(
            peft_config.base_model_name_or_path,
            trust_remote_code=True,
            cache_dir=model_args.cache_dir,
        )
        # from model config to model class
        architecture = model_config.architectures[0].lower()
        if any(x in architecture for x in ["gpt", "causal"]):
            pipeline_type = "text-generation"
            model_class = AutoModelForCausalLM
        elif any(x in architecture for x in ["encoder-decoder", "seq2seq"]):
            pipeline_type = "text2text-generation"
            model_class = AutoModelForSeq2SeqLM
        else:
            raise ValueError(f"Unsupported architecture {architecture} for PEFT model.")

        model = model_class.from_pretrained(
            peft_config.base_model_name_or_path,
            config=model_config,
            trust_remote_code=True,
            quantization_config=bnb_config,
            cache_dir=model_args.cache_dir,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(
            model,
            model_args.peft_name_or_path,
            subfolder=model_args.peft_name_or_path_subfolder,
            is_trainable=False,
            device_map="auto",
            cache_dir=model_args.cache_dir,
        )
    else:
        model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True,
            cache_dir=model_args.cache_dir,
        )
        # from model config to model class
        architecture = model_config.architectures[0].lower()
        if any(x in architecture for x in ["gpt", "causal"]):
            pipeline_type = "text-generation"
            model_class = AutoModelForCausalLM
        elif any(
            x in architecture
            for x in [
                "encoder-decoder",
                "seq2seq",
                "t5",
                "bart",
                "conditionalgeneration",
            ]
        ):
            pipeline_type = "text2text-generation"
            model_class = AutoModelForSeq2SeqLM
        else:
            raise ValueError(f"Unsupported architecture {architecture}")

        model = model_class.from_pretrained(
            model_args.model_name_or_path,
            subfolder=model_args.model_name_or_path_subfolder,
            config=model_config,
            trust_remote_code=True,
            quantization_config=bnb_config,
            cache_dir=model_args.cache_dir,
            device_map="auto",
        )
    model.eval()

    return (
        pipeline(pipeline_type, model=model, tokenizer=tokenizer),
        pipeline_type,
    )
