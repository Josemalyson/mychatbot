import os
import tempfile

import streamlit as st
from dotenv import load_dotenv
from langchain import hub
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

load_dotenv()
model = os.getenv('MODEL')
openai_api_key = os.getenv('OPENAI_API_KEY')
humbugging_api_token = os.getenv('HUGGINGFACEHUB_API_TOKEN')
repo_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
docs = [Document(page_content=' none ')]
history = StreamlitChatMessageHistory(key="chat_messages")
chat_history = []
# App title
st.set_page_config(page_title="🦙💬 Llama 2 Chatbot")

with st.sidebar:
    st.title('🦙💬 Llama 2 Chatbot')
    st.subheader('Models and parameters')
    selected_model = st.sidebar.selectbox('Choose a model', ['Gemma', 'OpenAI', 'Mistral'],
                                          key='selected_model')

    temperature = st.sidebar.slider('temperature', min_value=0.1, max_value=1.0, value=0.1, step=0.1)

    if selected_model == 'Gemma':
        llm = ChatOllama(model='neural-chat', base_url='http://localhost:11434',
                         callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
                         , temperature=temperature)
        prompt = hub.pull("rlm/rag-prompt-llama")

    elif selected_model == 'OpenAI':
        llm = ChatOpenAI(temperature=temperature, model=model, openai_api_key=openai_api_key, streaming=True)
        prompt = hub.pull("rlm/rag-prompt")
    elif selected_model == 'Mistral':
        callbacks = [streaming_stdout.StreamingStdOutCallbackHandler()]
        llm = HuggingFaceEndpoint(
            repo_id=repo_id, temperature=temperature,
            huggingfacehub_api_token=humbugging_api_token, callbacks=callbacks,
        )
        prompt = hub.pull("rlm/rag-prompt-mistral")

    uploaded_url = st.text_input("Choose Site URL")
    uploaded_youtube_url = st.text_input("Choose Youtube URL")
    uploaded_file = st.file_uploader("Choose a file")
    # image = st.file_uploader("Choose a Image")

    st.markdown(
        '📖 Learn how to build this app in this [blog](https://blog.streamlit.io/how-to-build-a-llama-2-chatbot/)!')

# # Store LLM generated responses
# if "messages" not in st.session_state.keys():
#     st.session_state.messages = [{"role": "assistant", "content": "Fala ai Zé o que tu quer hoje ?"}]
#
# # Display or clear chat messages
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])

if len(history.messages) == 0:
    history.add_ai_message("Fala ai Zé o que tu quer hoje ?")

for msg in history.messages:
    st.chat_message(msg.type).write(msg.content)


def format_docs(data):
    return "\n\n".join(doc.page_content for doc in data)


def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "Fala ai Zé o que tu quer hoje ?"}]
    streamlit_js_eval(js_expressions="parent.window.location.reload()")


st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)
contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]


if question := st.chat_input():
    st.chat_message("human").write(question)

    if uploaded_url is not None and uploaded_url != '':
        loader = UnstructuredURLLoader(urls=[uploaded_url])
        docs = loader.load()

    if uploaded_youtube_url is not None and uploaded_youtube_url != '':
        loader = YoutubeLoader.from_youtube_url(
            uploaded_youtube_url,
            add_video_info=True,
            language=["pt"],
            translation="pt",
        )
        docs = loader.load()

    if uploaded_file is not None and uploaded_file != '':
        # Create tmp file
        temp_dir = tempfile.TemporaryDirectory()
        temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
        with open(temp_filepath, "wb") as f:
            f.write(uploaded_file.getvalue())
        loader = UnstructuredFileLoader(temp_filepath)
        docs = loader.load()

    # if image is not None and image != '':
    #     # Create tmp file
    #     temp_dir = tempfile.TemporaryDirectory()
    #     temp_filepath = os.path.join(temp_dir.name, image.name)
    #     with open(temp_filepath, "wb") as f:
    #         f.write(image.getvalue())
    #     loader = UnstructuredImageLoader(temp_filepath, mode="elements")
    #     docs = loader.load()

    print(docs)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(docs)

    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\

    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )

    rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever | format_docs
            )
            | prompt
            | llm
    )

    chain_with_history = RunnableWithMessageHistory(
        rag_chain,
        lambda session_id: history,  # Always return the instance created earlier
        input_messages_key="question",
        history_messages_key="chat_history",
    )

    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": question}, config)
    if response is not None and hasattr(response, 'content'):
        st.chat_message("ai").write(response.content)
    else:
        st.chat_message("ai").write(response)

    vectorstore.delete_collection()

# def rag(question, retrievers):
#     template = """Você é um assistente para tarefas de resposta a perguntas. Use as seguintes partes do contexto
#     recuperado para responder à pergunta. Se você não sabe a resposta, basta dizer que não sabe. Use frases e
#     mantenha a resposta concisa.
#
#     Question: {question}
#
#     Context: {context}
#
#     Answer:
#     """
#
#     prompt_template = PromptTemplate(input_variables=['question', 'context'], output_parser=None, partial_variables={},
#                                      template=template,
#                                      template_format='f-string', validate_template=True)
#
#     # prompt default in https://smith.langchain.com/hub
#     # prompt_template = hub.pull("rlm/rag-prompt")
#
#     rag_chain = (
#             {"context": retrievers | format_docs, "question": RunnablePassthrough()}
#             | prompt_template
#             | llm
#             | StrOutputParser()
#     )
#
#     return rag_chain.invoke(question)

#
# # User-provided prompt
# if prompt := st.chat_input():
#     st.session_state.messages.append({"role": "user", "content": prompt})
#     with st.chat_message("user"):
#         st.write(prompt)
#
# if st.session_state.messages[-1]["role"] != "assistant":
#     with st.chat_message("assistant"):
#         with st.spinner("Thinking..."):
#
#             if uploaded_url is not None and uploaded_url != '':
#                 loader = UnstructuredURLLoader(urls=[uploaded_url])
#                 docs = loader.load()
#
#             if uploaded_youtube_url is not None and uploaded_youtube_url != '':
#                 loader = YoutubeLoader.from_youtube_url(
#                     uploaded_youtube_url,
#                     add_video_info=True,
#                     language=["pt"],
#                     translation="pt",
#                 )
#                 docs = loader.load()
#
#             if uploaded_file is not None and uploaded_file != '':
#                 # Create tmp file
#                 temp_dir = tempfile.TemporaryDirectory()
#                 temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
#                 with open(temp_filepath, "wb") as f:
#                     f.write(uploaded_file.getvalue())
#                 loader = UnstructuredFileLoader(temp_filepath)
#                 docs = loader.load()
#
#             # if image is not None and image != '':
#             #     # Create tmp file
#             #     temp_dir = tempfile.TemporaryDirectory()
#             #     temp_filepath = os.path.join(temp_dir.name, image.name)
#             #     with open(temp_filepath, "wb") as f:
#             #         f.write(image.getvalue())
#             #     loader = UnstructuredImageLoader(temp_filepath, mode="elements")
#             #     docs = loader.load()
#
#             print(docs)
#
#             text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#             splits = text_splitter.split_documents(docs)
#
#             vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
#             retriever = vectorstore.as_retriever()
#
#             response = rag(prompt, retriever)
#             vectorstore.delete_collection()
#
#         st.write(response)
#     st.session_state.messages.append({"role": "assistant", "content": response})
