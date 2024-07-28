# source: https://github.com/huggingface/transformers/blob/08a2edfc6629a323effd7a85feafed9e6701e2dd/examples/pytorch/text-classification/run_glue.py#L56
task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "all": (),
    "vnrte": ("sentence1", "sentence2"),
    "vsfc": ("sentence", None),
    "vsmec": ("sentence", None),
    "vtoc": ("sentence", None),
    "wnli": ("sentence1", "sentence2"),
}
