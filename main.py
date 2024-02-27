import os
import time
from io import StringIO

import streamlit as st
from dotenv import load_dotenv
from langchain.callbacks import streaming_stdout
from langchain.chains import LLMChain
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.ollama import Ollama
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from openai import OpenAI
from streamlit_js_eval import streamlit_js_eval

load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
model = os.getenv('MODEL')
client = OpenAI(api_key=openai_api_key)
huggingfacehub_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')

repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"  # See https://huggingface.co/models?pipeline_tag=text-generation&sort=downloads for some other options

# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a Llama2 model', ['Llama2', 'OpenAI', 'Mistral'],
                                          key='selected_model')

    temperature = st.sidebar.slider('temperature', min_value=0.1, max_value=1.0, value=0.1, step=0.1)

    if selected_model == 'Llama2':
        llm = Ollama(model='llama2', base_url='http://localhost:11434',
                     callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                     , temperature=temperature)

    elif selected_model == 'OpenAI':
        llm = ChatOpenAI(temperature=temperature, model=model, openai_api_key=openai_api_key, streaming=True)
    elif selected_model == 'Mistral':
        callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
        llm = HuggingFaceEndpoint(
            repo_id=repo_id, temperature=temperature,
            huggingfacehub_api_token=huggingfacehub_api_token, callbacks=callbacks,
        )

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


def response_ollama(question):
    template = """

    <<SYS>> Você é um assistente pessoal de AI. <</SYS>>
    [INST] Responda todas as perguntas de forma simples e objetivo no idoma português. Ao final de cada resposta adicione o modelo
        que você é exemplos: gpt-turbo, ollama e entre outros.
        {question} 
    [/INST]

    """

    prompt_template = PromptTemplate(input_variables=['question'], output_parser=None, partial_variables={},
                                     template=template,
                                     template_format='f-string', validate_template=True)

    llm_chain = LLMChain(prompt=prompt_template, llm=llm)
    response_llm = llm_chain.run({"question": question})

    for word in response_llm.split():
        yield word + " "
        time.sleep(0.05)


def response_chatgpt():
    # default chat without context
    stream = client.chat.completions.create(
        model=model,
        messages=[
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.messages
        ],
        stream=True,
    )
    return st.write_stream(stream)


def response_huggingface(question):
    return llm(question)


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

            if selected_model == 'Llama2':
                response = response_ollama(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
                message = {"role": "assistant", "content": full_response}

            elif selected_model == 'OpenAI':
                response = response_chatgpt()
                message = {"role": "assistant", "content": response}

            elif selected_model == 'Mistral':
                response = response_huggingface(prompt)
                message = {"role": "assistant", "content": response}
                st.write(response)

    st.session_state.messages.append(message)
