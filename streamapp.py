import streamlit as st
import os
import time
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up API keys
groq_api_key = os.getenv('GROQ_API_KEY')
inference_api_key = os.getenv('HUGGING_FACE_TOKEN')

# Initialize the LLM and embeddings
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Llama3-70b-8192", temperature=0)
hfembeddings = HuggingFaceInferenceAPIEmbeddings(api_key=inference_api_key, model_name="BAAI/bge-large-en-v1.5")


vectorstore = FAISS.load_local("faiss_index", hfembeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# Set up prompts and chains
contextualize_q_system_prompt = (
    """Given a chat history and the latest user question, which might reference context from the chat history, 
    formulate a standalone question that can be understood without the need for the chat history. 
    Do NOT answer the question; simply reformulate it if needed and otherwise return it as is, 
    particularly focusing on the context of Jessup Cellars and the role of a Product and Sales Manager at the Jessup Cellars Wine Company."""
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

system_prompt = (
    """You are an assistant for question-answering tasks related to the operations, culture, history, wines, 
    wine-making processes, people of Jessup Cellars and many more. Use only the context to answer the question. 
    The context is the Jessup Cellars Corpus and provides all details related to Jessup Cellars. 
    Act like a Product and Sales Manager.Give clear and concise but correct answers.
    If asked anything outside the context, say 'Contact Business directly for more information.'
    \n\n{context}"""
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

conversational_rag_chain = RunnableWithMessageHistory(
    rag_chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
    output_messages_key="answer",
)

# Streamlit UI
st.title("Jessup Cellars Chatbot")

# Initialize session state variables
if "session_id" not in st.session_state:
    st.session_state["session_id"] = "abc12"
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

# User input
user_input = st.text_input("Enter your question:")

if user_input:
    start_time = time.process_time()
    answer = conversational_rag_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": st.session_state["session_id"]}},
    )
    st.session_state["chat_history"].append(f"🙎‍♂️You: {user_input}")
    st.session_state["chat_history"].append(f"🤖Bot: {answer['answer']}")
    response_time = time.process_time() - start_time

    st.write(f"Response time: {response_time:.2f} seconds")

# Display chat history
for chat in st.session_state["chat_history"]:
    st.write(chat)

# Clear chat button
if st.button("Clear Chat"):
    st.session_state["chat_history"] = []
