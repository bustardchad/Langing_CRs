"""Python file to serve as the frontend"""
import streamlit as st
from streamlit_chat import message
from streamlit_extras.colored_header import colored_header
from streamlit_extras.add_vertical_space import add_vertical_space

from langchain.chains import ConversationChain

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.vectorstores import FAISS

import arxiv
import os

from Reading_PDFs import *

# Get embeddings 
embeddings = HuggingFaceEmbeddings()


def load_chain(texts,embeddings):
    """Logic for loading the chain you want to use should go here."""
    #llm = OpenAI(temperature=0)
    #chain = ConversationChain(llm=llm)

    # Which LLM to use...
    repo_id = "google/flan-t5-base"
    #repo_id = "bigscience/bloom-1b1"

    print("Loading LLM " + str(repo_id))
    llm_hf = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.1, "max_length":64})

    print(f"Getting embeddings")
    #docsearch, qa = get_embeddings(texts, embeddings, llm_hf)
    chain = get_embeddings_ver2(texts, embeddings, llm_hf)

    return chain

# From here down is all the StreamLit UI.
st.set_page_config(page_title="arxiv-agent", page_icon=":robot:")
st.header("arxiv-agent")

# Sidebar contents
with st.sidebar:
    st.title('arxiv-agent')
    st.markdown('''
    ## About
    This app is an LLM-powered "agent" created with:
    - [Streamlit](https://streamlit.io/)
    - [LangChain]
    - [HuggingFace]
    
    Its purpose is to help you query the arxiv for papers and respond to questions about those papers.
    ''')
    add_vertical_space(5)
    st.write('Lovingly made by Chad Bustard, slightly competent overlord of the arxiv-agent')

if "generated" not in st.session_state:
    st.session_state["generated"] = []

if "past" not in st.session_state:
    st.session_state["past"] = []

def answer_questions(user_input):
    query = user_input
    docs = docsearch.similarity_search(query)
    
    # for testing sake, use docs[0].page_content as output
    #st.write("Here's the document data with the highest relevance to your prompt: \n ")
    output = docs[0].page_content
    #print(docs[0].page_content)
    #print(f"Answering query '{query}'")

    #output = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
    #output = chain.run(input=user_input)

    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)

def get_question():
    input_text = st.text_input("Enter a question...I'll give you the most relevant part of the document: ", " ", key="input")
    return input_text



# First query to user -- what keywords to search for on arxiv
good_search = False
#while (good_search==False):
search_query = st.text_input("Type some keywords, and I'll find the most relevant papers: ", " ", key="search_input")

# We will query and store urls in a list
url_list = []

#search_query = "cosmic ray streaming"
# if user inputs a search query, write out and search top 1 results sorted by relevancy
if search_query:
    search = arxiv.Search(
        query = search_query,
        max_results = 1,
        sort_by = arxiv.SortCriterion.Relevance
        #sort_by = arxiv.SortCriterion.SubmittedDate
    )

    # Load PDFs
    #st.write("Querying arxiv and loading most relevant paper")
    ct = 1
    for result in search.results():
        st.write(str(ct) + '--' + result.title)
        st.write(result.pdf_url)
        url_list.append(result.pdf_url)
        url = result.pdf_url
        ct += 1

    if(st.button('Learn about this paper')):

        # Load selected PDF
        texts = load_one_pdf(url)

        # load chain that gets vector embeddings from document and returns Q/A functionality
        st.write("Learning about this paper...(might take a few minutes)...")
        chain = load_chain(texts,embeddings)

        # Use FAISS to create docsearch, which allows us to do similarity search, etc.
        docsearch = FAISS.from_documents(texts,embeddings)

        # Prompt user for further questions about queried documents
        user_input = get_question()
        if user_input:
            docs = docsearch.similarity_search(user_input)
            
            # for testing sake, use docs[0].page_content as output
            #st.write("Here's the document data with the highest relevance to your prompt: \n ")
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
