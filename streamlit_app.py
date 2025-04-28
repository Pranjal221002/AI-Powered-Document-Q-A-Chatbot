import os
import streamlit as st
from chatbot import process_documents, create_qa_chain, get_answer

st.markdown(
    """
    <style>
        .stApp {
            background-color: #007BFF;  /* Blue background */
            color: white;  /* White text for contrast */
        }
        .main-title { 
            text-align: center; 
            font-size: 2.5em; 
            font-weight: bold; 
            color: #ffffff; 
        }
    </style>
    """,
    unsafe_allow_html=True
)


st.title("📄 AI Chatbot for Document Q&A")

# File uploader
uploaded_files = st.file_uploader(
    "Upload one or more documents (PDF, TXT, DOCX)", 
    type=["pdf", "txt", "docx"], 
    accept_multiple_files=True
)


if uploaded_files:
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)

    st.success("✅ Files uploaded successfully!")
    
    vectorstore = process_documents(file_paths)
    qa_chain, prompt_template = create_qa_chain(vectorstore)

    query = st.text_input("💬 Ask a question about the documents:")
    
    if st.button("Get Answer") and query:
        response, sources, tokens, cost = get_answer(qa_chain, prompt_template, query)

        st.write("### 📝 Answer:", response)
        st.write(f"📊 Tokens Used: {tokens}")
        st.write(f"💰 Cost: ${cost:.5f}")

        if sources:
            st.write("#### 📌 Sources:")
            for doc in sources:
                st.write(f"- {doc.metadata.get('source', 'Unknown Source')}")

