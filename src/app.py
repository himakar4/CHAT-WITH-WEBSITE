import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

def get_response(user_input):
    return "I dont know"

#app config
st.set_page_config(page_title="Chat with websites", page_icon="*")
st.title("Chat with websites")
chat_history = {
    AIMessage(content="Hello, I am a bot, How can i help you ?")
}

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")


#user input
user_query =  st.chat_input("Type your message here.. ")
if user_query is not None and user_query !="":
    response=get_response(user_query)
    chat_history.append(HumanMessage(content=user_query))
    # with st.chat_message("Human"):
    #     st.write(user_query)
    # with st.chat_message("AI"):
    #     st.write(response)



