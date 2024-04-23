import torch
from models.gpt_model import load_model_and_tokenizer
from data.data_loader import load_and_prepare_data, get_dataset_splits, tokenize_datasets
from utils.data_processing import collate_fn
from scripts.train import train_model, setup_device
from scripts.evaluate import evaluate_model, calculate_perplexity
from torch.utils.data import DataLoader
from torch.optim import AdamW
from scripts.generate import setup_generator, generate_responses


def main():
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    # Load and prepare datasets
    dataset = load_and_prepare_data()
    train_dataset, valid_dataset, test_dataset = get_dataset_splits(dataset)

    # Tokenize datasets
    tokenized_train, tokenized_valid, tokenized_test = tokenize_datasets(train_dataset, valid_dataset, test_dataset, tokenizer)

    # Setup device
    device = setup_device()

    # Create DataLoaders for each dataset split
    train_loader = DataLoader(tokenized_train, batch_size=2, shuffle=True, collate_fn=collate_fn, num_workers=2)
    valid_loader = DataLoader(tokenized_valid, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2)
    test_loader = DataLoader(tokenized_test, batch_size=2, shuffle=False, collate_fn=collate_fn, num_workers=2)

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # Train the model
    train_model(model, train_loader, device, optimizer, num_epochs=5)

    # Evaluate the model
    evaluate_model(model, valid_loader, device)

    #Save the model
    model_save_path = '/content/distilgpt2_chatbot/final_project/saved_models/model_weights.pth'
    directory = os.path.dirname(model_save_path)

    if not os.path.exists(directory):
        os.makedirs(directory)

    # Now save the model
    torch.save(model.state_dict(), model_save_path)


    # Setup generator and generate responses
    generator = setup_generator(device)
    # Assume `test_dataset` is already loaded and available
    generate_responses(generator, test_dataset)

    # Calculate perplexity
    calculate_perplexity(model, test_loader, device)


if __name__ == "__main__":
    main()
