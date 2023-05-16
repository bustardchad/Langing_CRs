"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message

from langchain.chains import ConversationChain

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import FAISS

import arxiv


from Reading_PDFs import *

def load_chain(texts,embeddings):
    """Logic for loading the chain you want to use should go here."""
    #llm = OpenAI(temperature=0)
    #chain = ConversationChain(llm=llm)

    # Which LLM to use...
    repo_id = "google/flan-t5-base"
    #repo_id = "bigscience/bloom-1b1"

    print("Loading LLM " + str(repo_id))
    llm_hf = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.1, "max_length":20})

    print(f"Getting embeddings")
    #docsearch, qa = get_embeddings(texts, embeddings, llm_hf)
    chain = get_embeddings_ver2(texts, embeddings, llm_hf)

    return chain

# From here down is all the StreamLit UI.
st.set_page_config(page_title="LangChain Demo", page_icon=":robot:")
st.header("LangChain Demo")

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []


def get_text():
    input_text = st.text_input("What would you like to ask the all-knowing cosmic ray wizard?: ", " ", key="input")
    return input_text


embeddings = HuggingFaceEmbeddings()


# We will query and store urls in a list
url_list = []

search_query = "cosmic ray streaming"
print("Querying arxiv")
search = arxiv.Search(
    query = search_query,
    max_results = 5,
    sort_by = arxiv.SortCriterion.Relevance
    #sort_by = arxiv.SortCriterion.SubmittedDate
)

for result in search.results():
    print(result.title)
    print(result.pdf_url)
    url_list.append(result.pdf_url)

# load PDF of choice
url = url_list[0]

print("Loading PDF(s)")
texts = load_one_pdf(url)

chain = load_chain(texts,embeddings)

docsearch = FAISS.from_documents(texts,embeddings)

user_input = get_text()

if user_input:
    query = user_input
    docs = docsearch.similarity_search(query)
    
    # for testing sake, use docs[0].page_content as output
    output = docs[0].page_content
    #print(docs[0].page_content)
    #print(f"Answering query '{query}'")

    #output = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    #output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

if st.session_state["generated"]:

    for i in range(len(st.session_state["generated"]) - 1, -1, -1):
        message(st.session_state["generated"][i], key=str(i))
        message(st.session_state["past"][i], is_user=True, key=str(i) + "_user")
