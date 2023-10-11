import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.llms import CTransformers
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile

load_dotenv()

def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["Hello, Ask me anything"]
    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey !"]

def conversational_chat(query,chain,history):
    result = chain({"question": query, "chat_history": history})
    history.append((query,result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()
    with container:
        with st.form(key='my_form',clear_on_submit=True):
            user_input = st.text_input("Question:",placeholder="Ask about your Document", key="input")
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            with st.spinner("Generate response ......."):
                output = conversational_chat(user_input,chain,st.session_state["history"])
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)
    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state['past'][i],is_user=True, key=str(i) + "_user",avatar_style="thumbs")
                message(st.session_state['generated'][i],key=str(i),avatar_style="panda")

def create_conversational_chain(vector_store):
    load_dotenv()
    #create llm
    llm = Replicate(
        streaming=True,
        model= "meta/llama-2-70b:a52e56fee2269a78c9279800ec88898cecb6c8f1df22a6483132bea266648f00",
        callbacks=[StreamingStdOutCallbackHandler()],
        input={"temperature": 0.1, "max_length": 500,"top_p": 1}
    )

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type="stuff",
                                                  retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                  memory =memory)
    
    return chain


def main():
    initialize_session_state()
    st.title("Buddhist AI Chatbot")
    st.sidebar.title("Buddhist AI Chatbot Document Processing")
    uploaded_file = st.sidebar.file_uploader("Choose a file",accept_multiple_files=True)

    if uploaded_file:
        text = []
        for file in uploaded_file:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name
        loader = None
        if file_extension == '.pdf':
            loader = PyPDFLoader(temp_file_path)
        elif file_extension == '.docx' or file_extension == '.doc':
            loader = Docx2txtLoader(temp_file_path)
        elif file_extension == '.txt':
            loader = TextLoader(temp_file_path)
        if loader:
            text.extend(loader.load())
            os.remove(temp_file_path)


        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=2000,chunk_overlap=100,length_function=len)
        text_chunks = text_splitter.split_documents(text)

        #createe embeddings
        embeddings = HuggingFaceBgeEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2",
                                            model_kwargs={"device": "cpu"})
        #crete veco store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        chain = create_conversational_chain(vector_store)

        display_chat_history(chain)


if __name__ == "__main__":
    main()