import json


from dataclasses import dataclass, field, asdict
from typing import Optional, Dict

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, TrainingArguments
from transformers.utils.versions import require_version
from src.glue_utils import task_to_keys


MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


@dataclass
class BasePrintingArguments:
    def __str__(self):
        self_as_dict = asdict(self)

        self_as_dict = {
            k: f"<{k.upper()}>" if k.endswith("_token") else v
            for k, v in self_as_dict.items()
        }
        # return f"{self.__class__.__name__}" + json.dumps(self_as_dict, indent=2) + "\n"

        attrs_as_str = [f"    {k}={v},\n" for k, v in sorted(self_as_dict.items())]
        attrs_as_str = "".join(attrs_as_str)
        return f"{self.__class__.__name__}(\n{attrs_as_str})"

    def __repr__(self):
        return self.__str__()


@dataclass
class ViGLUEDataArguments(BasePrintingArguments):
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.
    """

    task_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the task to train on: "
            + ", ".join(task_to_keys.keys())
        },
    )
    dataset_name: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the dataset to use (via the datasets library)."},
    )
    dataset_config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "The configuration name of the dataset to use (via the datasets library)."
        },
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached preprocessed datasets or not."},
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to pad all samples to `max_seq_length`. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the training data."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the validation data."},
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "A csv or a json file containing the test data."},
    )

    def __post_init__(self):
        if self.task_name is not None:
            self.task_name = self.task_name.lower()
            self.task_names = self.task_name.split(",")
            for task in self.task_names:
                if task not in task_to_keys.keys():
                    raise ValueError(
                        f"Unknown task {task}, you should pick one in "
                        + ",".join(task_to_keys.keys())
                    )
        elif self.dataset_name is not None:
            pass
        elif self.train_file is None or self.validation_file is None:
            raise ValueError(
                "Need either a GLUE task, a training/validation file or a dataset name."
            )
        else:
            train_extension = self.train_file.split(".")[-1]
            assert train_extension in [
                "csv",
                "json",
            ], "`train_file` should be a csv or a json file."
            validation_extension = self.validation_file.split(".")[-1]
            assert (
                validation_extension == train_extension
            ), "`validation_file` should have the same extension (csv or json) as `train_file`."

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()


@dataclass
class ModelArguments(BasePrintingArguments):
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune, or train from scratch.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The model checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    model_name_or_path_subfolder: Optional[str] = field(default="")
    model_type: Optional[str] = field(
        default=None,
        metadata={
            "help": "If training from scratch, pass a model type from the list: "
            + ", ".join(MODEL_TYPES)
        },
    )
    config_overrides: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override some existing default config settings when a model is trained from scratch. Example: "
                "n_embd=10,resid_pdrop=0.2,scale_attn_weights=false,summary_type=cls_index"
            )
        },
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained config name or path if not the same as model_name"
        },
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "Pretrained tokenizer name or path if not the same as model_name"
        },
    )
    cache_dir: Optional[str] = field(
        default="cache",
        metadata={
            "help": "Where do you want to store the pretrained models downloaded from huggingface.co"
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."
        },
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."
        },
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    torch_dtype: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the "
                "dtype will be automatically derived from the model's weights."
            ),
            "choices": ["auto", "bfloat16", "float16", "float32"],
        },
    )
    low_cpu_mem_usage: bool = field(
        default=False,
        metadata={
            "help": (
                "It is an option to create the model as an empty shell, then only materialize its parameters when the pretrained weights are loaded."
                "set True will benefit LLM loading time and RAM consumption."
            )
        },
    )

    # peft parameters
    # peft type must be in one of the followi ["lora", "prefixtuning", "ptuning", "prompttuning", "ia3"] or None
    peft_type: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The type of PEFT to use. Choose from ['lora', 'prefixtuning', 'ptuning', 'prompttuning', 'ia3']"
            )
        },
    )
    peft_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path of the PEFT checkpoint for weights initialization. Don't set if you want to train a model from scratch."
            )
        },
    )
    peft_name_or_path_subfolder: Optional[str] = field(default="")
    peft_config_path: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The path of the PEFT config file. If not provided, will use the default PEFT config file."
            )
        },
    )

    # quantization parameters
    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": "Whether to convert the loaded model into mixed-8bit quantized model."
        },
    )
    load_in_4bit: bool = field(
        default=False,
        metadata={
            "help": "Whether to convert the loaded model into mixed-4bit quantized model."
        },
    )
    bnb_4bit_quant_type: str = field(
        default="fp4",
        metadata={
            "help": "bnb_4bit_quant_type (`str`, {fp4, nf4}, defaults to `fp4`):"
            " The quantization type of the model. Can be fp4 or nf4."
        },
    )
    bnb_4bit_compute_dtype: str = field(
        default="float32",
        metadata={
            "help": "The compute dtype of the model. Can be float32, fp32, float16, fp16"
            " bfloat16, bf16."
        },
    )
    bnb_4bit_use_double_quant: bool = field(
        default=False,
        metadata={"help": "Whether to use double quantization for the model."},
    )

    def __post_init__(self):
        if self.config_overrides is not None and (
            self.config_name is not None or self.model_name_or_path is not None
        ):
            raise ValueError(
                "--config_overrides can't be used in combination with --config_name or --model_name_or_path"
            )

        if self.model_name_or_path is None and self.peft_name_or_path is None:
            raise ValueError(
                "--model_name_or_path and --peft_name_or_path can't both be None"
            )

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()


@dataclass
class EvaluateArguments(BasePrintingArguments):
    prompt_type: str = field(
        default="en",
        metadata={"help": "The type of prompt to use. Choose from ['en', 'vi']"},
    )
    output_filename: str = field(
        default=None,
        metadata={
            "help": "The name of the output file to write the results to. If not provided, will use the name of the model."
        },
    )
    output_dir: str = field(
        default=None,
        metadata={
            "help": "The output directory where the results will be written.",
        },
    )
    debug: bool = field(
        default=False,
        metadata={
            "help": "Whether to run in debug mode (only compute for the first 3 batches for the first  template).",
        },
    )
    seed: int = field(
        default=42,
        metadata={
            "help": "Random seed to use for evaluation.",
        },
    )
    n_shots: str = field(
        default="0",
        metadata={
            "help": "Number of few-shot examples to use for each task.",
        },
    )

    evaluate_train: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to run on the train split.",
        },
    )
    evaluate_validation: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to run on the validation split.",
        },
    )
    evaluate_test: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to run on the test split.",
        },
    )

    batch_size: int = field(
        default=2,
        metadata={"help": "Batch size to use for evaluation."},
    )

    max_new_tokens: Optional[int] = field(
        default=10,
        metadata={"help": "Maximum number of new tokens to generate for each example."},
    )
    min_new_tokens: Optional[int] = field(
        default=None,
        metadata={"help": "Minimum number of new tokens to generate for each example."},
    )

    def __str__(self):
        return super().__str__()

    def __repr__(self):
        return super().__repr__()
