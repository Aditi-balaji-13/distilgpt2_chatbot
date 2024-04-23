import torch
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from models.gpt_model import load_model_and_tokenizer

# Disable parallelism in tokenizers to prevent issues in multiprocessing environments
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

tokenizer, model = load_model_and_tokenizer()

def setup_device():
    """Setup CUDA device if available, and clear cache."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda':
        torch.cuda.empty_cache()  # Clear unused memory
        total_memory = torch.cuda.get_device_properties(0).total_memory
        reserved_memory = torch.cuda.memory_reserved(0)
        free_memory = total_memory - reserved_memory
        required_memory = 256 * 1024 * 1024  # Assume model needs 256MB
        print(f"Total GPU memory: {total_memory}, Reserved memory: {reserved_memory}, Free memory: {free_memory}, Required memory: {required_memory}")

        if free_memory >= required_memory:
            print("Sufficient GPU memory available, using GPU.")
        else:
            print("Not enough GPU memory, switching to CPU.")
            device = torch.device("cpu")
    else:
        print("Using CPU.")
    return device

def train_model(model, train_loader, device, optimizer, num_epochs=5):
    model.to(device)
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'labels'}
            labels = batch['labels'].to(device)
            optimizer.zero_grad()
            outputs = model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {total_loss / len(train_loader)}')
