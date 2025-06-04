import torch
from models.gpt_model import load_model_and_tokenizer

# Load model and tokenizer
tokenizer, model = load_model_and_tokenizer()

def evaluate_model(model, eval_loader, device):
    model.eval()
    total_eval_loss = 0
    with torch.no_grad():
        for batch in eval_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_eval_loss += loss.item()
    avg_loss = total_eval_loss / len(eval_loader)
    print(f'Validation Loss: {avg_loss}')
    return avg_loss

def calculate_perplexity(model, test_loader, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()

    # Calculate perplexity based on total loss
    perplexity = torch.exp(torch.tensor(total_loss / len(test_loader)))
    print(f"Perplexity: {perplexity}")
    return perplexity
