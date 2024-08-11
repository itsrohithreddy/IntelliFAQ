import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
# from langchain_community.llms import HuggingFaceHub
# from langchain_community.llms import HuggingFacePipeline
from langchain.chains.question_answering import load_qa_chain
from langchain_text_splitters import RecursiveCharacterTextSplitter
# from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
# import google.generativeai as genai
# genai.configure(api_key='AIzaSyCkgmLEmBnQj3HqaF8M4dmVPoBdlvLte3Q')
# model = genai.GenerativeModel('gemini-1.0-pro')

# model_id = "google/gemma-2b"
# tokenizer = AutoTokenizer.from_pretrained(model_id, token="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")
# model = AutoModelForCausalLM.from_pretrained(model_id, token="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")
os.environ['GOOGLE_API_KEY']='AIzaSyCkgmLEmBnQj3HqaF8M4dmVPoBdlvLte3Q'
def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text

def get_text_chunks(text):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=400)
  chunks = text_splitter.split_text(text)
  return chunks

def get_vector_store(text_chunks):
  vector_store = Chroma.from_texts(
      texts=text_chunks,
      embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'),
      persist_directory="./chroma_db"
  )

def get_conversational_chain():
  prompt_template = """
  Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
  provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
  Context:\n {context}?\n
  Question: \n{question}\n

  Answer:
  """

  prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
  # pipe = pipeline(
  #     "text-generation", model=model,tokenizer=tokenizer, max_new_tokens=100,device = 0
  # )
  # hf = HuggingFacePipeline(pipeline=pipe)
  model = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)

  chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
  return chain

def user_input(user_question):
  new_db = Chroma(persist_directory="./chroma_db", embedding_function = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2'))
  docs = new_db.similarity_search(user_question,k=3)
  chain = get_conversational_chain()
  response = chain(
      {"input_documents":docs, "question": user_question}
      ,return_only_outputs=True)
  st.write("Reply: ", response["output_text"])

def main():
  st.set_page_config("ChatBot with custom PDF")
  st.header("Chat with PDF")

  user_question = st.text_input("Ask a Question from the PDF Files")

  if user_question:
      user_input(user_question)

  with st.sidebar:
      st.title("Menu:")
      pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
      if st.button("Submit & Process"):
          with st.spinner("Processing..."):
              raw_text = get_pdf_text(pdf_docs)
              text_chunks = get_text_chunks(raw_text)
              get_vector_store(text_chunks)
              st.success("Done")



if __name__ == "__main__":
  main()