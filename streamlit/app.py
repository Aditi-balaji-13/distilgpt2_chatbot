import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextGenerationPipeline
import streamlit as st

@st.cache(allow_output_mutation=True)
def load_model_and_tokenizer():
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    model = AutoModelForCausalLM.from_pretrained("distilgpt2", config={"pad_token_id": tokenizer.eos_token_id})

    # Load saved model weights from the specified path
    model_load_path = '/content/distilgpt2_chatbot/final_project/saved_models/model_weights.pth'
    model.load_state_dict(torch.load(model_load_path, map_location=torch.device('cpu')))
    model.eval()  # Set model to evaluation mode

    return tokenizer, model

def generate_response(model, tokenizer, context, max_length, temperature):
    generator = TextGenerationPipeline(
        model=model,
        tokenizer=tokenizer,
        device=-1 if not torch.cuda.is_available() else 0,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        do_sample=True,
        num_return_sequences=1,
        temperature=temperature
    )
    print(f"Context: {context}")  # Debug print
    generated_responses = generator(context)
    response = generated_responses[0]['generated_text']
    print(f"Generated response: {response}")  # Debug print
    return response


def main():
    st.title("Chatbot Interface")

    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()

    # Sidebar for parameter adjustments
    with st.sidebar:
        st.header("Adjust Response Settings")
        temperature = st.slider("Temperature", min_value=0.01, max_value=1.0, value=0.7, step=0.01)
        max_length = st.slider("Max Length", min_value=20, max_value=200, value=100, step=10)

    user_input = st.text_input("You:", "")

    if st.button("Generate Response"):
        if user_input:  # Ensure there is input to process
            response = generate_response(model, tokenizer, user_input, max_length, temperature)
            st.text_area("Chatbot:", response, height=300)
        else:
            st.write("Please enter some text to chat about.")

if __name__ == "__main__":
    main()
