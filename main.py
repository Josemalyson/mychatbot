import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import UnstructuredURLLoader, YoutubeLoader, UnstructuredFileLoader
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain_community.vectorstores.chroma import Chroma
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler, streaming_stdout
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from streamlit_js_eval import streamlit_js_eval

# load envs
load_dotenv()

# envs
model = os.getenv('MODEL')
openai_api_key = os.getenv('OPENAI_API_KEY')
humbugging_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
repo_id = os.getenv('REPO_ID')
model_ollama = os.getenv('MODEL_OLLAMA')
chunk_size = os.getenv('CHUNK_SIZE')
chunk_overlap = os.getenv('CHUNK_OVERLAP')

docs = [Document(page_content=' none ')]
history = StreamlitChatMessageHistory(key="chat_history")
chat_history = []

# create tittle
st.set_page_config(page_title="💻 🖱️ Zé Chatbot")


def create_llm():
    if selected_model == 'OpenAI':
        return ChatOpenAI(temperature=temperature, model=model, openai_api_key=openai_api_key, streaming=True)

    elif selected_model == 'Nvidia - HF':
        callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
        return HuggingFaceEndpoint(
            repo_id=repo_id, temperature=temperature,
            huggingfacehub_api_token=humbugging_api_token, callbacks=callbacks,
        )

    elif selected_model == 'Llama2 - Local':
        return ChatOllama(model=model_ollama, base_url='http://localhost:11434',
                          callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                          , temperature=temperature)


# create sidebar

with st.sidebar:
    # create title
    st.title('💻 🖱️ Zé Chatbot')
    st.subheader('Models and parameters')

    # create select
    selected_model = st.sidebar.selectbox('Choose a model', ['OpenAI', 'Nvidia - HF', 'Llama2 - Local'],
                                          key='selected_model')

    # config temperratue
    temperature = st.sidebar.slider('temperature', min_value=0.1, max_value=1.0, value=0.1, step=0.1)

    # config model llm
    llm = create_llm()

    # create components upload
    site_url = st.text_input("Choose Site URL")
    youtube_url = st.text_input("Choose Youtube URL")
    uploaded_file = st.file_uploader("Choose a file")

    st.markdown(
        '📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')


# clean and reload page
def clear_chat_history():
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


# format documents in text format
def format_docs(data):
    return "\n\n".join(doc.page_content for doc in data)


# create default prompt with chat history
contextualize_q_system_prompt = """Dado um histórico de bate-papo e a última pergunta do usuário \
que pode fazer referência ao contexto no histórico do bate-papo, formule uma pergunta independente \
que pode ser entendido sem o histórico de bate-papo. NÃO responda à pergunta, \
apenas reformule-o se necessário e devolva-o como está. responda sempre em português do Brasil."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# create chain with LCE
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

# create Load, chunk and index the contents
if site_url is not None and site_url != '':
    loader = UnstructuredURLLoader(urls=[site_url])
    docs = loader.load()
    expander = st.expander("Contexto Adicionado")
    expander.write(format_docs(docs))

# create Load, chunk and index the contents
if youtube_url is not None and youtube_url != '':
    loader = YoutubeLoader.from_youtube_url(
        youtube_url,
        add_video_info=True,
        language=["pt"],
        translation="pt",
    )
    docs = loader.load()
    expander = st.expander("Contexto Adicionado")
    expander.write(format_docs(docs))

# create Load, chunk and index the contents
if uploaded_file is not None and uploaded_file != '':
    # Create tmp file
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = UnstructuredFileLoader(temp_filepath)
    docs = loader.load()
    expander = st.expander("Contexto Adicionado")
    expander.write(format_docs(docs))

# show messages chat
for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)


# logic to call prompt default without context
def contextualized_question(data: dict):
    if data.get("chat_history"):
        return contextualize_q_chain
    else:
        return data["question"]


# get question of user
if question := st.chat_input():
    st.chat_message("human").write(question)

    # create SPLIT
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    # Store splits
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

    # create STORE
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    # create template with context
    qa_system_prompt = """Você é um assistente para tarefas de resposta a perguntas. \
     Use as seguintes partes do contexto recuperado para responder à pergunta. \
     Se você não sabe a resposta, basta dizer que não sabe. \
     Use no máximo três frases e mantenha a resposta concisa.  responda sempre em português do Brasil.

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt), MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    # RetrievalQA
    rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever | format_docs
            )
            | qa_prompt
            | llm
            | StrOutputParser()
    )

    # Chain add chat history
    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: history,
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    config = {"configurable": {"session_id": "any"}}

    # chain with stream active
    answer = chain_with_history.stream({"question": question}, config)

    # choice response to difference models of llm
    if answer is not None and hasattr(answer, 'content'):
        st.chat_message("ai").write_stream(answer.content)
    else:
        st.chat_message("ai").write_stream(answer)

    # clean vector database
    vectorstore.delete_collection()
