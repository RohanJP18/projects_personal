import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

PDF_FILE_PATH = "Physcs pdf.pdf"

def get_pdf_text(pdf_path):
    text = ""
    pdf_reader = PdfReader(pdf_path)
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are an expert physics tutor. Based on the provided context, generate a list of thoughtful and challenging questions that are strictly related to the specific concept the user has asked about. The questions should probe understanding of that particular concept and focus only on it. Provide a variety of questions, including multiple-choice, short-answer, and calculation-based questions, if applicable.
    Context: \n{context}\n
    User's requested concept: {concept}\n

    Create a list of physics questions based only on the concept provided:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "concept"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings=embeddings, allow_dangerous_deserialization=True)
    
    docs = new_db.similarity_search(user_question)

    context = "\n".join([doc.page_content for doc in docs])

    chain = get_conversational_chain()

    response = chain.invoke({
        "input_documents": docs,
        "concept": user_question   
    }, return_only_outputs=True)

    st.write("Reply: ", response["output_text"])


def main():
    st.set_page_config("Chat PDF")
    st.header("Make any physics question")

    user_question = st.text_input("Ask a Question about Physics: \n")

    if user_question:
        raw_text = get_pdf_text(PDF_FILE_PATH)
        print(raw_text)
        text_chunks = get_text_chunks(raw_text)
        print(text_chunks)
        get_vector_store(text_chunks)
        user_input(user_question)
        st.success("Done")

if __name__ == "__main__":
    main()
