from datasets import load_dataset
from utils.data_processing import tokenize_function


def load_and_prepare_data():
    print("Loading dataset...")
    dataset = load_dataset("databricks/databricks-dolly-15k")
    print("Dataset loaded:", dataset)
    # Inspect the structure of the dataset, e.g., columns and splits
    print("Splits in the dataset:", dataset.keys())
    print("Columns in the train split:", dataset['train'].column_names)
    return dataset


def get_dataset_splits(dataset):
    print("Splitting dataset...")
    # Split the training data to create new validation and test sets
    train_testvalid = dataset['train'].train_test_split(test_size=0.2) # Reserve 20% of the data for testing and validation
    test_valid = train_testvalid['test'].train_test_split(test_size=0.5) # Split the reserved 20% equally into test and validation sets
    train_dataset = train_testvalid['train'] # Remaining 80% for training
    valid_dataset = test_valid['test'] # 10% of the original dataset for validation
    test_dataset = test_valid['train'] # 10% of the original dataset for testing
    print(f"Training set size: {len(train_dataset)}")
    print(f"Validation set size: {len(valid_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    return train_dataset, valid_dataset, test_dataset


def tokenize_datasets(train_dataset, valid_dataset, test_dataset, tokenizer):
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_valid = valid_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    tokenized_test = test_dataset.map(lambda examples: tokenize_function(examples, tokenizer), batched=True)
    return tokenized_train, tokenized_valid, tokenized_test
