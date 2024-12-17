import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings.spacy_embeddings import SpacyEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.agents import AgentExecutor, create_tool_calling_agent


load_dotenv()


embedding_model = SpacyEmbeddings(model_name="en_core_web_sm")

def extract_text_from_pdfs(pdf_files):
 
 extracted_text = ""
 for pdf in pdf_files:
 reader = PdfReader(pdf)
 for page in reader.pages:
 extracted_text += page.extract_text()
 return extracted_text

def split_text_into_chunks(text):
 
 splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
 return splitter.split_text(text)

def create_vector_database(chunks):
 
 vector_db = FAISS.from_texts(chunks, embedding=embedding_model)
 vector_db.save_local("vector_database")

def generate_conversation_chain(retrieval_tool, user_query):
 
 
 language_model = ChatAnthropic(
 model="claude-3-sonnet-20240229",
 temperature=0,
 api_key=os.getenv("ANTHROPIC_API_KEY"),
 verbose=True
 )

 
 chat_prompt = ChatPromptTemplate.from_messages([
 ("system", """You are an intelligent assistant. Respond with detailed answers using the provided context. 
 If you cannot find the answer within the given information, reply with: 'Answer not located in the provided context.'. Avoid making incorrect assumptions."""),
 ("placeholder", "{chat_history}"),
 ("human", "{input}"),
 ("placeholder", "{agent_scratchpad}")
 ])

 
 agent = create_tool_calling_agent(language_model, [retrieval_tool], chat_prompt)
 executor = AgentExecutor(agent=agent, tools=[retrieval_tool], verbose=True)

 
 result = executor.invoke({"input": user_query})
 st.write("Response:", result['output'])

def handle_user_query(query):
 vector_db = FAISS.load_local("vector_database", embedding_model, allow_dangerous_deserialization=True)
 retriever = vector_db.as_retriever()
 retrieval_tool = create_retriever_tool(retriever, "document_retrieval", "Tool for answering PDF-based queries.")
 generate_conversation_chain(retrieval_tool, query)

def main():
 
 st.set_page_config(page_title="PDF Chat Assistant")
 st.header("PDF-Based Question Answering System")

 query = st.text_input("Enter your question related to the uploaded PDFs:")
 if query:
 handle_user_query(query)

 
 with st.sidebar:
 st.title("Options:")
 updfs = st.file_uploader("Here you Add PDF documents ", accept_multiple_files=True)
 if st.button("Process Files"):
 with st.spinner("Extracting and processing data..."):
 if updfs:
 raw_text = extract_text_from_pdfs(updfs)
 text_chunks = split_text_into_chunks(raw_text)
 create_vector_database(text_chunks)
 st.success("PDFs processed successfully!")
 else:
 st.warning("Please make sure to upload a PDF file at a minimum.")

if __name__ == "__main__":
 main()
