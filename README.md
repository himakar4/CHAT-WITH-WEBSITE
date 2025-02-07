# Chat with Websites from URL - LangChain Chatbot using Groq AI with Streamlit GUI
Welcome to the GitHub repository for the LangChain Chatbot using Groq AI with Streamlit GUI! This project is a comprehensive guide to building a chatbot capable of interacting with websites, extracting information, and communicating in a user-friendly manner. It leverages the power of LangChain and integrates it with a Streamlit GUI for an enhanced user experience. The Groq API keys are free to use and for generating embeddings Pinecone also offers limited usage of its resources.
# Features
- Website Interaction: The chatbot uses the latest version of LangChain to interact with and extract information from various websites.
- Large Language Model Integration: Compatibility with models like GPT-4, Llama3, and ollama, Groq, MistralAI. In this code I am using Groq chat model, but you can change it to any other model.
- Embeddings & Vector Store: I have used Pinecone for embeddings and ChromaDB for vector store.
- Streamlit GUI: A clean and intuitive user interface built with Streamlit, making it accessible for users with varying levels of technical expertise.
- Python-based: Entirely coded in Python.

# Brief explanation of how RAG works
A RAG bot is short for Retrieval-Augmented Generation. This means that we are going to "augment" the knowledge of our LLM with new information that we are going to pass in our prompt. We first vectorize all the text that we want to use as "augmented knowledge" and then look through the vectorized text to find the most similar text to our prompt. We then pass this text to our LLM as a prefix.
