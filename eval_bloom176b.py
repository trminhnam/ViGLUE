import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import os
import warnings
import json
import warnings
import time
import requests
import random

import torch
import evaluate
from datasets import load_dataset
from tqdm.auto import tqdm

from transformers import HfArgumentParser, set_seed

from src.arguments import ViGLUEDataArguments, ModelArguments, EvaluateArguments

from promptsource.templates import Template
from datasets.utils.logging import disable_progress_bar

disable_progress_bar()
warnings.filterwarnings("ignore")

ALL_GLUE_SUBSETS = [
    "cola",
    "mnli",
    "mrpc",
    "qnli",
    "qqp",
    "rte",
    "sst2",
    "vnrte",
    "vsfc",
    "vsmec",
    "vtoc",
    "wnli",
]

SPLIT_2_EVAL = {
    "ax": [],
    "cola": ["train", "validation"],
    "mnli": ["train", "validation_matched", "validation_mismatched"],
    "mrpc": ["train", "validation", "test"],
    "qnli": ["train", "validation"],
    "qqp": ["train", "validation"],
    "rte": ["train", "validation"],
    "sst2": ["train", "validation"],
    "vnrte": ["train", "validation"],
    "vsfc": ["train", "validation", "test"],
    "vsmec": ["train", "validation", "test"],
    "vtoc": ["train", "validation"],
    "wnli": ["train", "validation"],
}


class CustomCollator:
    def __init__(
        self,
        template,
        fewshot_samples=None,
        debug=False,
    ):
        self.template = template
        self.fewshot_samples = fewshot_samples
        self.debug = debug

    def __call__(self, batch):
        if self.debug:
            print(batch)

        instructions = []
        labels = []
        for i in batch:
            instruction, label = task_template.apply(
                {**i, "examples": self.fewshot_samples}
            )
            instructions.append(instruction)
            labels.append(label)

            if self.debug:
                print(instruction)
                print(label)
                print()

        return {
            "instructions": instructions,
            "labels": labels,
        }


def format_name_for_saving(name):
    return name.replace("/", "_").replace(" ", "_").strip()


def postprocess_for_sc(preds, labels, label_classes):
    # preprocess
    preds = [p.strip().lower() for p in preds]
    labels = [l.strip().lower() for l in labels]
    label_classes = [l.strip().lower() for l in label_classes]

    # get the mapping from index to label and vice versa
    id_to_class = {i: label_classes[i] for i in range(len(label_classes))}
    class_to_id = {v: k for (k, v) in id_to_class.items()}

    # get the index of the true label
    references = [class_to_id[d] for d in labels]

    # get the index of the predicted label
    predictions = []
    label_classes = sorted(label_classes, key=len, reverse=True)
    for pred, label in zip(preds, labels):
        # find the class that is in the prediction
        for i, c in enumerate(label_classes):
            if c in pred:
                predictions.append(i)
                break
        else:
            # the prediction does not contain any class
            # select the class that not equal to the label
            for i, c in enumerate(label_classes):
                if c != label:
                    predictions.append(i)
                    break
            else:
                predictions.append(-1)
    return predictions, references


if __name__ == "__main__":
    global_start_time = time.time()
    parser = HfArgumentParser((ModelArguments, ViGLUEDataArguments, EvaluateArguments))
    model_args, data_args, eval_args = parser.parse_args_into_dataclasses()

    running_time = time.time()

    ### SEED
    set_seed(eval_args.seed)

    ### OUTPUT_DIR
    if eval_args.output_dir is None:
        eval_args.output_dir = "./eval_results"
    os.makedirs(eval_args.output_dir, exist_ok=True)

    if eval_args.output_filename is not None:
        save_file_name = eval_args.output_filename
    else:
        if model_args.peft_name_or_path:
            save_file_name = f"{format_name_for_saving(model_args.peft_name_or_path)}_{data_args.dataset_name.split('/')[-1]}_{data_args.task_name}_{eval_args.prompt_type}.json"
        else:
            save_file_name = f"{format_name_for_saving(model_args.model_name_or_path)}_{data_args.dataset_name.split('/')[-1]}_{data_args.task_name}_{eval_args.prompt_type}.json"
    save_file_path = os.path.join(eval_args.output_dir, save_file_name)

    print(
        model_args.__str__() + "\n" + data_args.__repr__() + "\n" + eval_args.__repr__()
    )

    ### LOAD PROMPTS for all tasks
    prompt_template_path = f"templates/prompt_collections.{eval_args.prompt_type}.json"
    all_template_data = json.load(open(prompt_template_path, "r", encoding="utf-8"))
    all_fewshot_data = json.load(
        open(f"resources/fewshot_samples.json", "r", encoding="utf-8")
    )

    n_shots = eval_args.n_shots
    if n_shots == -1:
        n_shots = [0, 1, 2, 4, 8, 16]
    else:
        n_shots = [int(x) for x in n_shots.split(",")]

    if hasattr(data_args, "task_name") and data_args.task_name is not None:
        if data_args.task_name == "all":
            run_all_tasks = True
            tasks = ALL_GLUE_SUBSETS
        else:
            run_all_tasks = False
            tasks = data_args.task_name.split(",")
    else:
        run_all_tasks = True
        tasks = ALL_GLUE_SUBSETS

    results = {}
    if os.path.exists(save_file_path):
        print(f"Loading results from {save_file_path}")
        results = json.load(open(save_file_path, "r", encoding="utf-8"))

    API_URL = "https://api-inference.huggingface.co/models/bigscience/bloom"
    HF_TOKEN = os.getenv("HF_TOKEN")

    def cal_api(prompt):
        def query(payload):
            token = HF_TOKEN
            headers = {"Authorization": f"Bearer {token}"}
            response = requests.post(API_URL, headers=headers, json=payload)
            return response.json()

        def get_model_inputs(prompt):
            return {
                "inputs": prompt,
                "parameters": {
                    "do_sample": False,
                    "top_k": 1,
                    "wait_for_model	": True,
                    "return_full_text": False,
                    "max_new_tokens": eval_args.max_new_tokens,
                },
            }

        return query(get_model_inputs(prompt))

    for task_name in tasks:
        dataset = load_dataset(
            data_args.dataset_name,
            task_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True,
        )
        task_template = Template(**all_template_data[task_name])
        if task_name not in results:
            results[task_name] = {}

        ### Evaluate
        for split in SPLIT_2_EVAL[task_name]:
            if "train" in split and not eval_args.evaluate_train:
                continue
            if "validation" in split and not eval_args.evaluate_validation:
                continue
            if "test" in split and not eval_args.evaluate_test:
                continue

            if "validation_mismatched" in split:
                continue

            if split not in results[task_name]:
                results[task_name][split] = {}

            for shot in n_shots:
                start_time = time.time()
                print(
                    f"*** Evaluate {data_args.dataset_name}/{task_name}/{split} ({shot} shot) ***"
                )
                if data_args.dataset_name in {"glue", "tmnam20/ViGLUE"}:
                    if task_name in {"vnrte", "vsmec", "vsfc", "vtoc"}:
                        metric = evaluate.load("glue", "rte")
                    else:
                        metric = evaluate.load("glue", task_name)
                else:
                    if task_name in {"vnrte", "vsmec", "vsfc", "vtoc"}:
                        metric = evaluate.load("glue", "rte")
                    else:
                        metric = evaluate.load(data_args.dataset_name, task_name)

                collator = CustomCollator(
                    task_template,
                    fewshot_samples=all_fewshot_data[task_name][:shot],
                    debug=eval_args.debug,
                )

                dataloader = torch.utils.data.DataLoader(
                    dataset[split],
                    batch_size=eval_args.batch_size,
                    shuffle=False,
                    num_workers=0,
                    collate_fn=collator,
                )

                pbar = tqdm(
                    dataloader,
                    desc=f"Evaluate {data_args.dataset_name}/{task_name} ({shot} shot)",
                    total=len(dataloader),
                )
                for i, batch in enumerate(pbar):
                    generated_contents = []
                    for prompt in batch["instructions"]:
                        n_tries = 10
                        while n_tries > 0:
                            try:
                                response = cal_api(prompt)
                                generated_contents.append(response[0]["generated_text"])
                                break
                            except:
                                print(response)
                                print(f"Error: {n_tries} tries left")
                                if n_tries == 0:
                                    raise Exception("API failed")
                                n_tries -= 1
                                time.sleep(random.randint(60, 120))

                    preds, refs = postprocess_for_sc(
                        generated_contents,
                        batch["labels"],
                        task_template.get_fixed_answer_choices_list(),
                    )

                    metric.add_batch(
                        predictions=preds,
                        references=refs,
                    )

                # average metric over all templates
                split_metric_result = metric.compute()
                print(split_metric_result)
                print(f"Time: {time.time() - start_time:.2f}s")
                results[task_name][split][shot] = split_metric_result
                print()
                with open(os.path.join(save_file_path), "w", encoding="utf-8") as f:
                    json.dump(results, f, indent=4, ensure_ascii=False)

    print(json.dumps(results, indent=4, ensure_ascii=False))
    print(f"Total tinme: {time.time() - global_start_time:.2f}s")

    with open(os.path.join(save_file_path), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print(f"Saved results to {save_file_path}")
