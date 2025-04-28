import os
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Function to load files
def load_file(file_path):
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    elif file_path.endswith(".txt"):
        loader = TextLoader(file_path)
    elif file_path.endswith(".docx"):
        loader = Docx2txtLoader(file_path)
    else:
        raise ValueError("Unsupported file format")
    return loader.load()

# Process document & create vectorstore
def process_documents(file_paths):
    all_documents = []
    for file_path in file_paths:
        documents = load_file(file_path)
        all_documents.extend(documents)

    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_documents(all_documents)

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    return vectorstore


# Create QA chain
def create_qa_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-4o-mini", openai_api_key=OPENAI_API_KEY)
    retriever = vectorstore.as_retriever()

    # Custom Prompt Template
    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="You are an AI assistant that answers questions based on the provided document.\n\nContext: {context}\n\nQuestion: {question}\n\nAnswer:",
    )
    
    qa_chain = RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever,return_source_documents=True)
    return qa_chain, prompt_template

# Function to handle query
def get_answer(qa_chain, prompt_template, query):
    with get_openai_callback() as cb:
        formatted_prompt = prompt_template.format(context="Relevant document sections", question=query)
        result = qa_chain.invoke({"query": formatted_prompt})
        
        response = result["result"]
        sources = result["source_documents"]

        return response, sources, cb.total_tokens, cb.total_cost
