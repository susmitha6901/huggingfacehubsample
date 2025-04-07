import os
import faiss
from PyPDF2 import PdfReader
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEndpoint

def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

def create_vectorstore_from_pdf(file_paths, api_key):
    all_text = ""
    for file_path in file_paths:
        all_text += load_pdf(file_path)

    splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(all_text)

    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    sample_embedding = embedding_model.embed_query("sample")
    dimension = len(sample_embedding)

    index = faiss.IndexFlatL2(dimension)
    vectorstore = FAISS(
        embedding_function=embedding_model.embed_query,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )
    vectorstore.add_texts(chunks)
    return vectorstore

def get_answer(vectorstore, question, api_key):
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a helpful assistant. Use the following context from a PDF document to answer the user's question.

Context:
{context}

Question:
{question}

Only use the above context. If you don't know the answer, say "I couldn't find the answer in the PDF."

Note: NO Need to mention "Answer/Response such type of Keywords
Note: Generate the Response in Just 2-3 lines and keep the format very clean and clear"
"""
    )

    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3-8B-Instruct",
        temperature=0.6,
        huggingfacehub_api_token=api_key
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain.run(question)
