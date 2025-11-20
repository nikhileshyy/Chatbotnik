import streamlit as st
from PyPDF2 import PdfReader
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit import sidebar
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
Open_API_Key=""
st.header("NoteBot")
with sidebar:
    st.title("My Notes")
    file=st.file_uploader("upload notes pdf",type="pdf")
#extracting from the file
if file is not None:
    my_pdf=PdfReader(file)
    text=""
    for page in my_pdf.pages:
        text += page.extract_text()
        st.write(text)
#break it into chunks
    splitter=RecursiveCharacterTextSplitter(chunk_size=300,chunk_overlap=50)
    chunks=splitter.split_text(text)
    #st.write(chunks)
    #creating objects of openAIEmbedding's class that let us connect with openAIEmbedding models
    embeddings=OpenAIEmbeddings(api_key=Open_API_Key)
    FAISS.from_texts(chunks,embeddings)

    #creating vectorstore and storing embeddings in it
    vector_store=FAISS.from_texts(chunks,embeddings)

    #getuserquery
    user_query=st.text_input("type your query")

    #semanticsearch from vector store
    if user_query:
        matching_chunks=vector_store.similarity_search(user_query)


        #define our LLM
        llm=ChatOpenAI(
            api_key=Open_API_Key,
            max_tokens=200,
            temperature=0,
        )
#generate response
        chain=load_qa_chain(llm,chain_type="stuff")
        output=chain.run(question=user_query,input_documents=matching_chunks)
        st.write(output)
