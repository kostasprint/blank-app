import streamlit as st
import os
import requests
import json
from transformers import AutoTokenizer

# App title
st.set_page_config(page_title="WeMake Streamlit OpenRouter", page_icon="💬", layout="wide")

# OpenRouter Credentials
with st.sidebar:
    st.title('💬 WeMake Streamlit OpenRouter ')
    st.write('Models hosted at [OpenRouter](https://openrouter.ai/).')
    if 'OPENROUTER_API_KEY' in st.secrets:
        openrouter_api_key = st.secrets['OPENROUTER_API_KEY']
    else:
        openrouter_api_key = st.text_input('Enter OpenRouter API key:', type='password')
        if not openrouter_api_key:
            st.warning('Please enter your OpenRouter API key.', icon='⚠️')
            st.markdown("**Don't have an API key?** Head over to [OpenRouter](https://openrouter.ai) to sign up for one.")

    # Create two columns in the sidebar
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Model 1")
        model1 = st.selectbox("Select model 1", ("openai/gpt-4o-mini","meta-llama/llama-3.2-90b-vision-instruct:free","anthropic/claude-3-haiku","google/gemini-flash-1.5","nvidia/llama-3.1-nemotron-70b-instruct", "qwen/qwen-2.5-72b-instruct"), key="model1")
        #temperature1 = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1, key="temp1")
        #max_tokens1 = st.number_input("Max Tokens", min_value=1, max_value=4096, value=1000, step=1, key="max_tokens1")
        temperature1 = 1.0
        max_tokens1 = 1000

    with col2:
        st.subheader("Model 2")
        model2 = st.selectbox("Select model 2", ("meta-llama/llama-3.2-90b-vision-instruct:free","anthropic/claude-3-haiku","google/gemini-flash-1.5","openai/gpt-4o-mini","nvidia/llama-3.1-nemotron-70b-instruct", "qwen/qwen-2.5-72b-instruct"), key="model2")
        #temperature2 = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0, step=0.1, key="temp2")
        #max_tokens2 = st.number_input("Max Tokens", min_value=1, max_value=4096, value=1000, step=1, key="max_tokens2")
        temperature2 = 1.0
        max_tokens2 = 1000
    
    st.divider()
    
    safe = True
    #safe = st.checkbox("Safe")

    # New text area for user instructions
    user_instructions = st.text_area("Instructions for the assistants:", "")

# Store LLM-generated responses
if "messages1" not in st.session_state.keys():
    st.session_state.messages1 = [{"role": "assistant", "content": "Ask me anything!"}]
if "messages2" not in st.session_state.keys():
    st.session_state.messages2 = [{"role": "assistant", "content": "Ask me anything!"}]

def clear_chat_history():
    st.session_state.messages1 = [{"role": "assistant", "content": "Ask me anything."}]
    st.session_state.messages2 = [{"role": "assistant", "content": "Ask me anything."}]

st.sidebar.button('Clear chat history', on_click=clear_chat_history)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)

if safe:
    safer = "VERY IMPORTANT: Be safeguarded. Assume the user is a kid, so tell her gently if she's acting inappropriately."
else:
    safer = ""

#my system instruction
my_system_instructions = "Use the language the user is using in the last prompt. Use only one language."

# Incorporate user instructions
system_instructions = f"{user_instructions} {safer} {my_system_instructions}".strip()

# Function for generating model response
def generate_response(prompt, model, temperature, max_tokens, messages):
    headers = {
        "Authorization": f"Bearer {openrouter_api_key}",
        "Content-Type": "application/json"
    }

    messages = [{"role": "system", "content": system_instructions}] + messages + [{"role": "user", "content": prompt}]

    data = {
        "model": model,
        "messages": messages,
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

# Create two columns
col1, col2 = st.columns(2)

# Display chat messages for both models
with col1:
    st.subheader("Model 1")
    for message in st.session_state.messages1:
        with st.chat_message(message["role"]):
            st.write(message["content"])

with col2:
    st.subheader("Model 2")
    for message in st.session_state.messages2:
        with st.chat_message(message["role"]):
            st.write(message["content"])

# User-provided prompt (shared between both models)
prompt = st.chat_input(disabled=not openrouter_api_key)

if prompt:
    st.session_state.messages1.append({"role": "user", "content": prompt})
    st.session_state.messages2.append({"role": "user", "content": prompt})
    
    # Display user message in both columns
    with col1:
        with st.chat_message("user"):
            st.write(prompt)
    
    with col2:
        with st.chat_message("user"):
            st.write(prompt)
    
    # Generate response for Model 1
    with col1:
        if st.session_state.messages1[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                response = generate_response(prompt, model1, temperature1, max_tokens1, st.session_state.messages1)
                full_response = st.write_stream(response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages1.append(message)
    
    # Generate response for Model 2 after Model 1 has finished
    with col2:
        if st.session_state.messages2[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                response = generate_response(prompt, model2, temperature2, max_tokens2, st.session_state.messages2)
                full_response = st.write_stream(response)
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages2.append(message)
