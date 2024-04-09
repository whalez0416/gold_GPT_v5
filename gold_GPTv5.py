#v2 프롬프터 업그레이드 | 멀티쿼리 업그레이드
#v3 캐시파일 구글 드라이브 연결
#v4 온라인 환경에서 사용할 수 있도록 구글 드라이브 키 연결
#v5 gpt 버전 업

from langchain.prompts import ChatPromptTemplate
from langchain.document_loaders import UnstructuredFileLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.callbacks.base import BaseCallbackHandler
import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.retrievers.multi_query import MultiQueryRetriever
from google.cloud import storage
import os
import json
from google.oauth2 import service_account


st.set_page_config(page_title="DocumentGPT", page_icon="📃")

# 환경 변수 검증 및 처리
def get_google_credentials():
    credentials_path = os.getenv('GOOGLE_APPLICATION_CREDENTIALS')
    if not credentials_path:
        raise ValueError("환경 변수가 설정되지 않았습니다.")
    try:
        with open(credentials_path, 'r') as file:
            service_account_info = json.load(file)
            return service_account.Credentials.from_service_account_info(service_account_info)
    except (json.JSONDecodeError, FileNotFoundError) as e:
        raise ValueError("서비스 계정 키 파일을 읽는 중 오류가 발생했습니다.") from e

credentials = get_google_credentials()
storage_client = storage.Client(credentials=credentials)

class ChatCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.message = ""
        self.message_box = None

    def on_llm_start(self, *args, **kwargs):
        if self.message_box is None:
            self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)

llm = ChatOpenAI(temperature=0.1, streaming=True, callbacks=[ChatCallbackHandler()])

def upload_to_gcs(bucket_name, source_file_name, destination_blob_name):
    try:
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_name)
        print(f"{source_file_name} has been uploaded to {bucket_name}/{destination_blob_name}")
    except Exception as e:
        print(f"Error uploading to GCS: {e}")
        raise



@st.cache_data(show_spinner="Embedding file...")
def embed_file(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}" 
    with open(file_path, "wb") as f:
        f.write(file_content)
    bucket_name = 'goldgpt_v2'  # GCS 버킷 이름
    upload_to_gcs(bucket_name, file_path, file.name)#캐시값을 자체 로컬파일 및 스트림릿에 업로드

    cache_dir = LocalFileStore(f"./.cache/embeddings/{file.name}")
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    embeddings = OpenAIEmbeddings()
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    multiquery_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), llm=llm
    )
    return multiquery_retriever


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            you are the head of professional consultation at the urology department. Our goal is to attract patients to our hospital by responding to all inquiries with the utmost kindness and warmth. you provide answers based only on the given context. If you are uncertain about a query, you kindly and warmly encourage the inquirer to contact the hospital directly for more information..          

            Additionally, to make patients feel more at ease, I use emoticons like the following:
            Example 1: Hello~ Good morning!
            Example 2: Thank you for your question ^^ The answer is as follows!

            If I have already 안녕하세요, I will not repeat 안녕하세요.
            
            Context: {context}
            """,
        ),
        ("human", "{question}"),
    ]
)


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = embed_file(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()
    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
            }
            | prompt
            | llm
        )
        with st.chat_message("ai"):
            chain.invoke(message)


else:
    st.session_state["messages"] = []

#pydantic.error_wrappers.ValidationError: This app has encountered an error. The original error message is redacted to prevent data leaks. Full error details have been recorded in the logs (if you're on Streamlit Cloud, click on 'Manage app' in the lower right of your app).

### 문제를 정리해보자 
# 지피티 말은 배포환경과 내 환경이 맞지 않아서 문제가 발생했다. 라는 건가 ?
# 리콰이얼먼츠 파일은 동일한 파일을 사용하고 있어 이건 맞나 ?
# 설사아니라고 해도 만약 아니였다면 로컬 환경에서 가상환경을 통해 실행했을때 문제가 됬지 않았을까 ??
# 캐시파일도 정상적으로 업로드 되었어 
# 실행은 문제가 없어###