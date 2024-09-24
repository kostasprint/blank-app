import streamlit as st
import replicate
import os
from transformers import AutoTokenizer

# App title
st.set_page_config(page_title="WeMake Streamlit Replicate Chatbot", page_icon="💬")

# Replicate Credentials
with st.sidebar:
    st.title('💬 WeMake Streamlit Replicate Chatbot')
    st.write('Create chatbots using various LLM models hosted at [Replicate](https://replicate.com/).')
    if 'REPLICATE_API_TOKEN' in st.secrets:
        replicate_api = st.secrets['REPLICATE_API_TOKEN']
    else:
        replicate_api = st.text_input('Enter Replicate API token:', type='password')
        if not (replicate_api.startswith('r8_') and len(replicate_api)==40):
            st.warning('Please enter your Replicate API token.', icon='⚠️')
            st.markdown("**Don't have an API token?** Head over to [Replicate](https://replicate.com) to sign up for one.")
    os.environ['REPLICATE_API_TOKEN'] = replicate_api

    st.subheader("Models and parameters")
    model = st.selectbox("Select a model",("meta/meta-llama-3-70b-instruct","meta/meta-llama-3.1-405b-instruct",), key="model")
    if model == "google-deepmind/gemma-2b-it":
        model = "google-deepmind/gemma-2b-it:dff94eaf770e1fc211e425a50b51baa8e4cac6c39ef074681f9e39d778773626"
    
    safe = st.sidebar.checkbox("Safe")
    temperature = 0.7
    top_p = 0.9

# qui le cose in eggs
    

# Store LLM-generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything Weila!."}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Ask me anything."}]

st.sidebar.button('Clear chat history', on_click=clear_chat_history)

@st.cache_resource(show_spinner=False)
def get_tokenizer():
    """Get a tokenizer to make sure we're not sending too much text
    text to the Model. Eventually we will replace this with ArcticTokenizer
    """
    return AutoTokenizer.from_pretrained("huggyllama/llama-7b")

def get_num_tokens(prompt):
    """Get the number of tokens in a given prompt"""
    tokenizer = get_tokenizer()
    tokens = tokenizer.tokenize(prompt)
    return len(tokens)



if safe:
    safer = "Always salute me with: ciao capo. Always reply to me in the language I'm speaking. VERY IMPORTANT: Be safeguarded. I'm a kid, tell me firmly if I'm acting inappropriately."
else:
    safer = " "
    

# Function for generating model response
def generate_response():
    prompt = []
#  <|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>
    prompt.append("<|im_start|>system<|im_end|>")
    prompt.append("\n\n{" + safer + "}<|im_end|>")
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            prompt.append("<|im_start|>user\n" + dict_message["content"] + "<|im_end|>")
        else:
            prompt.append("<|im_start|>assistant\n" + dict_message["content"] + "<|im_end|>")
    
    prompt.append("<|im_start|>assistant" + safer + "<|im_end|>assistant")
    prompt.append("")
    prompt_str = "\n".join(prompt)
    
    if get_num_tokens(prompt_str) >= 3072:
        st.error("Conversation length too long. Please keep it under 3072 tokens.")
        st.button('Clear chat history', on_click=clear_chat_history, key="clear_chat_history")
        st.stop()

    for event in replicate.stream(model,
                           input={"prompt": prompt_str,
                                  "prompt_template": r"{prompt}",
                                  "temperature": temperature,
                                  "top_p": top_p,
                                  }):
        yield str(event)

# User-provided prompt
if prompt := st.chat_input(disabled=not replicate_api):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        response = generate_response()
        full_response = st.write_stream(response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
