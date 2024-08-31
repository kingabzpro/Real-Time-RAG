import os
import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
import threading

# Load the API key from environment variables
groq_api_key = os.getenv("Groq_API_Key")

# Initialize the language model with the specified model and API key
llm = ChatGroq(model="llama-3.1-70b-versatile", api_key=groq_api_key)

# Initialize the embedding model
embed_model = HuggingFaceEmbeddings(
    model_name="mixedbread-ai/mxbai-embed-large-v1", model_kwargs={"device": "cpu"}
)

# Load the vector store from a local directory
vectorstore = Chroma(
    "Starwars_Vectordb",
    embedding_function=embed_model,
)

# Convert the vector store to a retriever
retriever = vectorstore.as_retriever()

# Define the prompt template for the language model
template = """You are a Star Wars assistant for answering questions. 
    Use the provided context to answer the question. 
    If you don't know the answer, say so. Explain your answer in detail. 
    Do not discuss the context in your response; just provide the answer directly.

    Context: {context}

    Question: {question}
    
    Answer:"""

rag_prompt = PromptTemplate.from_template(template)

# Create the RAG (Retrieval-Augmented Generation) chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | rag_prompt
    | llm
    | StrOutputParser()
)

# Global variable to store the current input text
current_text = ""

# Lock to synchronize access to current_text
text_lock = threading.Lock()


# Define the function to stream the RAG memory
def rag_memory_stream(text):
    global current_text
    with text_lock:
        current_text = text  # Update the current text input
    partial_text = ""
    for new_text in rag_chain.stream(text):
        with text_lock:
            # If the input text has changed, reset the generation
            if text != current_text:
                return  # Exit the generator if new input is detected
        partial_text += new_text
        # Yield the updated conversation history
        yield partial_text


# Set up the Gradio interface
title = "Real-time AI App with Groq API and LangChain"
description = """
<center>
<img src="https://huggingface.co/spaces/kingabzpro/Real-Time-RAG/resolve/main/Images/cover.png" alt="logo" width="550"/>
</center>
"""

demo = gr.Interface(
    title=title,
    description=description,
    fn=rag_memory_stream,
    inputs="text",
    outputs="text",
    live=True,
    batch=False,  # Disable batching to handle each input separately
    max_batch_size=1,  # Set batch size to 1 to process inputs one by one
    concurrency_limit=12,
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

# Launch the Gradio interface
demo.queue()
demo.launch()
