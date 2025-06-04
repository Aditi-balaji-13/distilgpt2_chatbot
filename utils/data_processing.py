import torch
from transformers import AutoTokenizer

def tokenize_function(examples, tokenizer, max_length=512):
    """ Tokenize inputs and labels from dataset examples. """
    tokenized_inputs = tokenizer(examples['context'], truncation=True, padding="max_length", max_length=max_length)
    tokenized_labels = tokenizer(examples['response'], truncation=True, padding="max_length", max_length=max_length)
    return {
        'input_ids': tokenized_inputs['input_ids'],
        'attention_mask': tokenized_inputs['attention_mask'],
        'labels': tokenized_labels['input_ids']
    }

def collate_fn(batch):
    """ Collate function to process data into batches for DataLoader. """
    input_ids = torch.stack([torch.tensor(item['input_ids'], dtype=torch.long) for item in batch])
    attention_mask = torch.stack([torch.tensor(item['attention_mask'], dtype=torch.long) for item in batch])
    labels = torch.stack([torch.tensor(item['labels'], dtype=torch.long) for item in batch])
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}
