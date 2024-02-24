from io import StringIO

import streamlit as st
from streamlit_js_eval import streamlit_js_eval

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

# Replicate Credentials
with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2', 'OpenAI', 'Mistral'],
                                          key='selected_model')
    if selected_model == 'Llama2':
        llm = ''
    elif selected_model == 'OpenAI':
        llm = ''
    elif selected_model == 'Mistral':
        llm = ''
    temperature = st.sidebar.slider('temperature', min_value=0.1, max_value=1.0, value=0.1, step=0.1)

    st.markdown(
        '📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Fala ai Zé o que tu quer hoje ?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Fala ai Zé o que tu quer hoje ?"}]
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# Function for generating LLaMA2 response. Refactored from https://github.com/a16z-infra/llama2-chatbot
def generate_llama2_response(prompt_input):
    string_dialogue = ("You are a helpful assistant. You do not respond as 'User' or pretend to be 'User'. You only "
                       "respond once as 'Assistant'.")
    for dict_message in st.session_state.messages:
        if dict_message["role"] == "user":
            string_dialogue += "User: " + dict_message["content"] + "\n\n"
        else:
            string_dialogue += "Assistant: " + dict_message["content"] + "\n\n"
    output = 'resposta'

    return output


uploaded_url = st.text_input("Choose Site URL")
uploaded_youtube_url = st.text_input("Choose Youtube URL")
uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    st.write(bytes_data)

    # To convert to a string based IO:
    stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
    st.write(stringio)

    # To read file as string:
    string_data = stringio.read()
    st.write(string_data)

if uploaded_url is not None:
    st.write(uploaded_url)

if uploaded_youtube_url is not None:
    st.write(uploaded_youtube_url)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama2_response(prompt)
            placeholder = st.empty()
            full_response = ''
            for item in response:
                full_response += item
                placeholder.markdown(full_response)
            placeholder.markdown(full_response)
    message = {"role": "assistant", "content": full_response}
    st.session_state.messages.append(message)
