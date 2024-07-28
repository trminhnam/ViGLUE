import json


def save_dataset_to_json(dataset, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)


def map_func_to_col(dataset, func, col):
    for i in range(len(dataset)):
        dataset[i][col] = func(dataset[i][col])
    return dataset
