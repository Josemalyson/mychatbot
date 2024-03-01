import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader, YoutubeLoader, UnstructuredFileLoader
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.llms.ollama import Ollama
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler, streaming_stdout
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit_js_eval import streamlit_js_eval

load_dotenv()
model = os.getenv('MODEL')
openai_api_key = os.getenv('OPENAI_API_KEY')
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

    uploaded_url = st.text_input("Choose Site URL")
    uploaded_youtube_url = st.text_input("Choose Youtube URL")
    uploaded_file = st.file_uploader("Choose a file")

    st.markdown(
        '📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "Fala ai Zé o que tu quer hoje ?"}]

# Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Fala ai Zé o que tu quer hoje ?"}]
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


def load_url_docs():
    # Load, chunk and index the contents of the blog.
    loader = UnstructuredURLLoader(urls=[uploaded_url])
    docs = loader.load()

    print(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    return vectorstore.as_retriever()


def load_url_youtube_docs():
    loader = YoutubeLoader.from_youtube_url(
        uploaded_youtube_url,
        add_video_info=True,
        language=["pt"],
        translation="pt",
    )
    docs = loader.load()

    print(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    return vectorstore.as_retriever()


def load_files():
    # Create tmp file
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = UnstructuredFileLoader(temp_filepath)
    docs = loader.load()

    print(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)
    # OllamaEmbeddings
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # Retrieve and generate using the relevant snippets of the blog.
    return vectorstore.as_retriever()


def response_ollama(question, retrievers):
    template = """

    <<SYS>>Você é um assistenten pessoal que responde no idioma português do Brasil as perguntas do usuário, 
        onde o mesmo enviaria uma questão e você entenderá o contexto enviado e com palavras simples explicará com 
        detalhes a reposta. <</SYS>>
    [INST]
        Question: {question} 
        
        Context: {context} 
        Answer: Responda com educação e de forma simples.

    [/INST]

    """

    prompt_template = PromptTemplate(input_variables=['question', 'context'], output_parser=None, partial_variables={},
                                     template=template,
                                     template_format='f-string', validate_template=True)

    rag_chain = (
            {"context": retrievers | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )

    response_llm = rag_chain.invoke(question)

    return response_llm


def response_chatgpt(question, retrievers):
    template = (
        """Você é um assistenten pessoal que responde no idioma português do Brasil as perguntas do usuário, 
        onde o mesmo enviaria uma questão e você entenderá o contexto enviado e com palavras simples explicará com 
        detalhes a reposta.
        
        Question: {question} 
        
        Context: {context} 
        
        Answer: Responda com educação e de forma simples.
        """

    )

    prompt_template = PromptTemplate(input_variables=['question', 'context'], output_parser=None, partial_variables={},
                                     template=template,
                                     template_format='f-string', validate_template=True)

    rag_chain = (
            {"context": retrievers | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )

    response_gtp = rag_chain.invoke(question)

    return response_gtp


def response_huggingface(question, retrievers):
    template = (
        """Você é um assistenten pessoal que responde no idioma português do Brasil as perguntas do usuário, 
        onde o mesmo enviaria uma questão e você entenderá o contexto enviado e com palavras simples explicará com 
        detalhes a reposta.

        Question: {question} 
        
        Context: {context} 
        
        Answer: Responda com educação e de forma simples.
        """

    )

    prompt_template = PromptTemplate(input_variables=['question', 'context'], output_parser=None, partial_variables={},
                                     template=template,
                                     template_format='f-string', validate_template=True)

    rag_chain = (
            {"context": retrievers | format_docs, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
    )

    response_hf = rag_chain.invoke(question)

    return response_hf


# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Generate a new response if last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):

            retriever = Chroma.from_documents(documents=[Document(page_content='')],
                                              embedding=OpenAIEmbeddings()).as_retriever()

            if uploaded_url is not None and uploaded_url != '':
                retriever = load_url_docs()

            if uploaded_youtube_url is not None and uploaded_youtube_url != '':
                retriever = load_url_youtube_docs()

            if uploaded_file is not None and uploaded_file != '':
                retriever = load_files()

            if selected_model == 'Llama2':
                response = response_ollama(prompt, retriever)
            elif selected_model == 'OpenAI':
                response = response_chatgpt(prompt, retriever)
            elif selected_model == 'Mistral':
                response = response_huggingface(prompt, retriever)

        st.write(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
