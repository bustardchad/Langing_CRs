from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os
import arxiv

#os.environ['HUGGINGFACEHUB_API_TOKEN'] = HUGGINGFACEHUB_API_TOKEN


#huggingfacehub_api_token = os.environ['HUGGINGFACEHUB_API_TOKEN']

from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import OnlinePDFLoader
from langchain.document_loaders import MathpixPDFLoader

from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain import VectorDBQA
from langchain.document_loaders import DirectoryLoader


def load_one_pdf(url):
    # Load in PDFs saved in url_list
    loader = PyPDFLoader(url)

    # PDF is my cosmic ray "drag" paper, Bustard and Oh 2023
    #loader = PyPDFLoader("https://arxiv.org/pdf/2301.04156.pdf")
    # pages = loader.load_and_split()
    
    #If you have this issue with e.g. the CharacterTextSplitter a work around 
    # is to use the RecursiveCharacterTextSplitter and set a couple of separators 
    #that work for you. This reduces the chance of having a chunk that doesn't fit

    #text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0, separators=[" ", ",", "\n"])
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=100, separators=[" ", ",", "\n"])
    documents = loader.load()
    texts = text_splitter.split_documents(documents)
    
    return texts


from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores import FAISS

def get_embeddings(texts, embeddings, llm):
    
    # need to set persist() in Jupyter notebook
    """
    persist_directory = 'persist_dir'
    docsearch = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
    docsearch.persist()
    docsearch = None
    
    # Now we can load the persisted database from disk, and use it as normal. 
    docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    
    docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    """
    docsearch = FAISS.from_documents(texts,embeddings)
    # Get documents relevant for the input query

    qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, 
                                                     chain_type="stuff", 
                                                     retriever=docsearch.as_retriever(),
                                                     max_tokens_limit=800,
                                                     reduce_k_below_max_tokens=True)
    return docsearch, qa
#timeout issue is documented here: https://huggingface.co/google/flan-t5-xxl/discussions/44



# using load
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
def get_embeddings_ver2(texts, embeddings, llm):
    
    template = """Given the following extracted parts of a long document and a question, create a final answer with references ("SOURCES"). 
    If you don't know the answer, just say that you don't know. Don't try to make up an answer.
    ALWAYS return a "SOURCES" part in your answer.
    Respond in English.

    QUESTION: {question}
    =========
    {summaries}
    =========
    FINAL ANSWER:"""
    PROMPT = PromptTemplate(template=template, input_variables=["summaries", "question"])

    qa = load_qa_with_sources_chain(llm, chain_type="stuff", prompt=PROMPT)
    
    return qa
#timeout issue is documented here: https://huggingface.co/google/flan-t5-xxl/discussions/44


# ## Putting it all together
# Let's now query the arxiv for recently submitted papers matching some keywords. We'll store the paper urls in a list and pass each url in a for loop through our pipeline of text splitting, embedding, and either querying or summarization

"""
from langchain.embeddings import HuggingFaceEmbeddings
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

# Which LLM to use...
repo_id = "google/flan-t5-base"
#repo_id = "bigscience/bloom-1b1"

print("Loading LLM " + str(repo_id))
llm_hf = HuggingFaceHub(repo_id=repo_id, model_kwargs={"temperature":0.1, "max_length":20})

print(f"Getting embeddings")
#docsearch, qa = get_embeddings(texts, embeddings, llm_hf)
docsearch, qa = get_embeddings_ver2(texts, embeddings, llm_hf)

#print(results["intermediate_steps"])

#print(result["source_documents"])



query = "What is cosmic ray streaming?"

docs = docsearch.similarity_search(query)
#print(docs[0].page_content)
print(f"Answering query '{query}'")
#docs = docsearch.get_relevant_documents(query)
results = qa({"input_documents": docs, "question": query}, return_only_outputs=True)

print(results)


# In[37]:


from langchain.chains.summarize import load_summarize_chain
#docs = docsearch.similarity_search(query)
chain = load_summarize_chain(llm_hf, chain_type="map_reduce")
docs = docsearch.similarity_search(query)
chain.run(docs)


# In[39]:


from langchain.chains import AnalyzeDocumentChain

summary_chain = load_summarize_chain(llm_hf, chain_type="map_reduce")
summarize_document_chain = AnalyzeDocumentChain(combine_docs_chain=summary_chain)

summarize_document_chain.run(texts)


# In[ ]:

"""


