import os,asyncio
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeEmbeddings
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

async def initialize_model():
    model = PineconeEmbeddings(model="multilingual-e5-large", api_key=PINECONE_API_KEY)
    return model

def get_vectorstore_from_url(url,model):
    # get the text in documents form
    loader = WebBaseLoader(url)

    # Disable SSL verification
    loader.requests_kwargs.update({'verify': False})
    document = loader.load()

    # split the document into chunks
    text_splitter= RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(document)

    #create a vectorstore from chunks
    vector_store = Chroma.from_documents(document_chunks, model )

    return vector_store 

def get_context_retriever_chain(vector_store):
    llm= ChatGroq(model="mixtral-8x7b-32768")

    retriever = vector_store.as_retriever()

    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the convo")
    ])

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    return retriever_chain

def get_conversational_rag_chain(retriever_chain):

    llm =ChatGroq(model="mixtral-8x7b-32768")

    prompt = ChatPromptTemplate([
        ("system", "Answer the user's questions based on the below context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user","{input}"),
    ])
    stuff_documents_chain = create_stuff_documents_chain(llm, prompt)

    return create_retrieval_chain(retriever_chain,stuff_documents_chain )


def get_response(user_input):
    #create conversation chain
    retriever_chain =get_context_retriever_chain(st.session_state.vector_store) 
    conversation_rag_chain= get_conversational_rag_chain(retriever_chain)

    response = conversation_rag_chain.invoke({
        "chat_history":st.session_state.chat_history,
        "input": user_query
    })

    return response['answer']


#app config
st.set_page_config(page_title="Chat with websites", page_icon="🤖")
st.title("Chat with websites")


# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")

if website_url is None or website_url=="":
    st.info("Please enter a website URL")

else :
    #session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
        AIMessage(content="Hello, How can i help you ?")
    ]
    if "vector_store" not in st.session_state:
        # Initialize model asynchronously
        model = initialize_model()
        st.session_state.vector_store = get_vectorstore_from_url(website_url, model)


    
    #user input
    user_query =  st.chat_input("Type your message here.. ")
    if user_query is not None and user_query !="":
        response=get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)



