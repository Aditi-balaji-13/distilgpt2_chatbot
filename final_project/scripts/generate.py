from transformers import TextGenerationPipeline
from models.gpt_model import load_model_and_tokenizer


def setup_generator(device):
    """Load the model and tokenizer, set up the text generation pipeline."""
    tokenizer, model = load_model_and_tokenizer()
    generator = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=device,  # Set this to -1 for CPU or 0 for GPU
        truncation=True,
        max_length=1500,
        do_sample=True,
        num_return_sequences=1
    )
    return generator


def generate_responses(generator, test_dataset):
    """Generate responses for test dataset using the generator."""
    for example in test_dataset.shuffle().select(range(10)):  # Adjust the range as needed
        prompt = example['context']
        generated_responses = generator(prompt)
        print(f"Context: {prompt}")
        print(f"Generated Response: {generated_responses[0]['generated_text']}")
        print(f"Actual Response: {example['response']}")
        print("-----")
