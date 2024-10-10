import streamlit as st
import os
import requests
import json
from transformers import AutoTokenizer

# App title
st.set_page_config(page_title="WeMake Streamlit OpenRouter Chatbot", page_icon="💬")

# OpenRouter Credentials
with st.sidebar:
    st.title('💬 WeMake Streamlit OpenRouter Chatbot')
    st.write('Create chatbots using various LLM models hosted at [OpenRouter](https://openrouter.ai/).')
    if 'OPENROUTER_API_KEY' in st.secrets:
        openrouter_api_key = st.secrets['OPENROUTER_API_KEY']
    else:
        openrouter_api_key = st.text_input('Enter OpenRouter API key:', type='password')
        if not openrouter_api_key:
            st.warning('Please enter your OpenRouter API key.', icon='⚠️')
            st.markdown("**Don't have an API key?** Head over to [OpenRouter](https://openrouter.ai) to sign up for one.")

    st.subheader("Models and parameters")
    model = st.selectbox("Select a model", ("openai/gpt-4o-mini-2024-07-18", "openai/gpt-3.5-turbo", "anthropic/claude-2", "google/palm-2-chat-bison"), key="model", index=0)
    
    temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)
    max_tokens = st.number_input("Max Tokens", min_value=1, max_value=4096, value=1000, step=1)

    # New selectbox for character choice
    character = st.selectbox("Choose a character", ("", "Snow White", "Pluto"), key="character")

    # New text area for user instructions
    user_instructions = st.text_area("Instructions for the assistant:", "")

# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything!"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything."}]

st.sidebar.button('Clear chat history', on_click=clear_chat_history)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

# Character instructions
character_instructions = f"Assume the role of {character} and respond accordingly." if character else ""

# Incorporate user instructions and character choice
system_instructions = f"{user_instructions} {character_instructions}".strip()

# Function for generating model response
def generate_response(prompt):
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }

    data = {
        "model": model,
        "messages": st.session_state.messages + [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True
    }

    response = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers=headers,
        json=data,
        stream=True
    )

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                line = line[6:]
                if line.strip() == '[DONE]':
                    break
                try:
                    json_object = json.loads(line)
                    content = json_object['choices'][0]['delta'].get('content', '')
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue

# User-provided prompt
if prompt := st.chat_input(disabled=not openrouter_api_key):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        full_response = st.write_stream(response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
