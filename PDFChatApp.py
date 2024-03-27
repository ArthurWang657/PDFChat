from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import  RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
import os
os.environ["OPENAI_API_KEY"] = "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"
file_path = "./林夏婷.pdf"
loader = PyPDFLoader(file_path)
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0) 
texts = loader.load_and_split(splitter)
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)
qa = ConversationalRetrievalChain.from_llm(ChatOpenAI(temperature=0), vectorstore.as_retriever())
chat_history = []
while True:
    query = input('\nQ: ') 
    if not query:
        break
    result = qa({"question": query + ' (用繁體中文回答)', "chat_history": chat_history})
    print('A:', result['answer'])
    chat_history.append((query, result['answer']))