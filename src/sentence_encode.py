from typing import List
import torch.nn.functional as F
from torch import Tensor
import torch
from transformers import AutoTokenizer, AutoModel


model_id = 'intfloat/multilingual-e5-base' # Model is unreliable
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModel.from_pretrained(model_id)
model.eval()

def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def encode_sentence(sentence: str) -> float:
    batch_dict = tokenizer([sentence], max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return F.normalize(embeddings, p=2, dim=1)[0]

def encode_sentences(sentences: List[str]) -> List[float]:
    batch_dict = tokenizer(sentences, max_length=512, padding=True, truncation=True, return_tensors='pt')
    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
    return F.normalize(embeddings, p=2, dim=1)


if __name__ == "__main__":
    print(torch.sum(encode_sentence('xin chào') * encode_sentence('hello')))
    print(torch.sum(encode_sentence('xin chào') * encode_sentence('tạm biệt')))

