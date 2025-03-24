
import os,asyncio
from dotenv import load_dotenv
import streamlit as st
from langchain_core.messages import AIMessage,HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


async def initialize_model():
    model = PineconeEmbeddings(model="multilingual-e5-large",pinecone_api_key=PINECONE_API_KEY)
    return model

async def get_vectorstore_from_url(url,model):
    
    # get text from url
    loader = WebBaseLoader(url)

    # Disable SSL verification
    loader.requests_kwargs.update({'verify': False})
    document = loader.load()

    # split the text into chunks
    text_spiltter = RecursiveCharacterTextSplitter()
    document_chunks = text_spiltter.split_documents(document)

    # storing the chunks into vectorstore
    vectorestore = Chroma.from_documents(document_chunks,model)

    return vectorestore


def get_context_retriever_chain(vector_store):
    llm = ChatGroq(model="llama-3.3-70b-specdec")

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name = "chat_history"),
        ("user",'{input}'),
        ("user","Given the convo")
    ])

    retriever_chain = create_history_aware_retriever(llm,retriever,prompt)

    return retriever_chain


def get_conversational_rag(retriever_chain):
    llm = ChatGroq(model="mixtral-8x7b-32768")

    prompt = ChatPromptTemplate.from_messages([
        ("system","Answer the users request based on the below context:\n\n {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user",'{input}'),
    ])

    stuffed_documnets_chain = create_stuff_documents_chain(llm,prompt)

    return create_retrieval_chain(retriever_chain,stuffed_documnets_chain)


def get_response(user_query):
    retriever_chain = get_context_retriever_chain(st.session_state.vector_store)
    conversation_rag_chain = get_conversational_rag(retriever_chain)
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    return response["answer"]


# Title 
st.set_page_config(page_title="Chat with Website",page_icon="🤖")
st.title("Chat with Websites")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Webiste URL")

if website_url is None or website_url == "":
    st.error("Please enter a website URL")
else:
    # chat history session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Welcome to the chat!")
        ]
    
    if "vector_store" not in st.session_state:
        # Initialize model asynchronously
        model = asyncio.run(initialize_model())
        st.session_state.vector_store = asyncio.run(get_vectorstore_from_url(website_url, model))
    
    # user input
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content = response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message,AIMessage):
            with st.chat_message("ai"):
                st.write(message.content)
        if isinstance(message,HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)
