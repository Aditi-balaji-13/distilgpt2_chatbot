from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model_and_tokenizer():
    print("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2")

    # Check if the tokenizer has an EOS token, and set pad token accordingly
    print("EOS Token:", tokenizer.eos_token)
    print("Current Pad Token:", tokenizer.pad_token)

    # Setting pad token
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.resize_token_embeddings(len(tokenizer))

    # Ensure padding on the left for decoder-only architectures
    tokenizer.padding_side = 'left'

    print("EOS Token:", tokenizer.eos_token)
    print("Current Pad Token:", tokenizer.pad_token)
    print("Padding side set to:", tokenizer.padding_side)

    return tokenizer, model
