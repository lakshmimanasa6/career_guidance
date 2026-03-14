import os
import streamlit as st
from typing import TypedDict, Annotated
from operator import add

from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import chromadb
from chromadb.utils import embedding_functions

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, AIMessage

from langgraph.graph import StateGraph, START, END



# ------------------------------------------------
# SET GROQ API KEY
# ------------------------------------------------

os.environ["GROQ_API_KEY"] = ""


# ------------------------------------------------
# INITIALIZE VECTOR DATABASE
# ------------------------------------------------

client = chromadb.Client()

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = client.get_or_create_collection(
    name="career_documents",
    embedding_function=embedding_function
)


# ------------------------------------------------
# INITIALIZE LLM
# ------------------------------------------------

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0
)


# ------------------------------------------------
# STATE DEFINITION
# ------------------------------------------------

class CareerState(TypedDict):
    query: str
    context: str
    use_retrieval: bool
    messages: Annotated[list, add]
    response: str


# ------------------------------------------------
# PDF INGESTION FUNCTION
# ------------------------------------------------

def ingest_pdf(file):

    reader = PdfReader(file)

    text = ""

    for page in reader.pages:

        page_text = page.extract_text()

        if page_text:
            text += page_text + "\n"

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_text(text)

    return chunks


# ------------------------------------------------
# DOCUMENT RETRIEVAL
# ------------------------------------------------

def retrieve_documents(query, top_k=3):

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    docs = results.get("documents", [])

    if docs:
        return docs[0]

    return []


# ------------------------------------------------
# NODE 1 : AGENT DECISION NODE
# LLM decides whether retrieval is required
# ------------------------------------------------

def agent_decision_node(state: CareerState):

    query = state["query"]

    prompt = f"""
You are an AI agent.

User Question:
{query}

Decide whether retrieving documents from a knowledge base is needed.

Reply ONLY with YES or NO.
"""

    decision = llm.invoke([HumanMessage(content=prompt)])

    use_retrieval = "YES" in decision.content.upper()

    return {
        **state,
        "use_retrieval": use_retrieval,
        "messages": [HumanMessage(content=query)]
    }


# ------------------------------------------------
# NODE 2 : RETRIEVAL TOOL NODE
# ------------------------------------------------

def retrieval_node(state: CareerState):

    query = state["query"]

    if state["use_retrieval"]:

        docs = retrieve_documents(query)

        context = "\n\n".join(docs)

    else:

        context = ""

    return {
        **state,
        "context": context
    }


# ------------------------------------------------
# NODE 3 : FINAL RESPONSE GENERATION
# ------------------------------------------------

def generate_response_node(state: CareerState):

    query = state["query"]

    context = state["context"]

    prompt = f"""
You are an AI Career Guidance Assistant.

Context:
{context}

User Question:
{query}

Provide clear and practical career guidance.
"""

    response = llm.invoke([HumanMessage(content=prompt)])

    return {
        **state,
        "response": response.content,
        "messages": state["messages"] + [AIMessage(content=response.content)]
    }


# ------------------------------------------------
# LANGGRAPH AGENT WORKFLOW
# ------------------------------------------------

workflow = StateGraph(CareerState)

workflow.add_node("agent_decision", agent_decision_node)
workflow.add_node("retrieval", retrieval_node)
workflow.add_node("generate_response", generate_response_node)

workflow.add_edge(START, "agent_decision")
workflow.add_edge("agent_decision", "retrieval")
workflow.add_edge("retrieval", "generate_response")
workflow.add_edge("generate_response", END)

app = workflow.compile()


# ------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------

st.title("🎓 Agentic AI Career Guidance Assistant")

st.write("Upload a **career-related PDF** and ask questions.")


# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:

    docs = ingest_pdf(uploaded_file)

    ids = [str(i) for i in range(len(docs))]

    collection.add(
        documents=docs,
        ids=ids
    )

    st.success("PDF processed and stored successfully!")


# User Query
query = st.text_input("Ask a career-related question")


if st.button("Get Career Advice"):

    if query.strip() == "":
        st.warning("Please enter a question.")

    else:

        initial_state = {
            "query": query,
            "context": "",
            "use_retrieval": False,
            "messages": [],
            "response": ""
        }

        result = app.invoke(initial_state)

        st.subheader("AI Career Advice")

        st.write(result["response"])