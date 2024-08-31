import os
import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate

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


# Define the function to stream the RAG memory
def rag_memory_stream(text, change_tracker):
    if change_tracker.get("changed", False):
        return  # Stop the generation if input has changed

    partial_text = ""
    for new_text in rag_chain.stream(text):
        if change_tracker.get("changed", False):
            return  # Stop the generation if input has changed
        partial_text += new_text
        yield partial_text  # Yield the updated conversation history


def input_listener(text, change_tracker):
    change_tracker["changed"] = True
    change_tracker["changed"] = False
    return text


# Initialize a change tracker
change_tracker = {"changed": False}

# Set up the Gradio interface
title = "Real-time AI App with Groq API and LangChain"
description = """
<center>
<img src="https://huggingface.co/spaces/kingabzpro/Real-Time-RAG/resolve/main/Images/cover.png" alt="logo" width="550"/>
</center>
"""

# Define input components with event listeners
text_input = gr.Textbox(label="Enter your question", elem_id="question")
text_input.change(
    fn=input_listener,
    inputs=[text_input],
    outputs=[text_input],
    change_tracker=change_tracker,
)

# Create the Gradio interface
demo = gr.Interface(
    title=title,
    description=description,
    fn=lambda text: rag_memory_stream(text, change_tracker),
    inputs=text_input,
    outputs="text",
    live=True,
    batch=True,
    max_batch_size=10000,
    concurrency_limit=12,
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

# Launch the Gradio interface
demo.queue()
demo.launch()