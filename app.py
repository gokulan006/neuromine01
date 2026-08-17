from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
import os 
from dotenv import load_dotenv
import streamlit as st
import pickle

from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
from langchain.agents import Tool, AgentExecutor, create_react_agent
 
import requests
import os
import zipfile

def download_from_drive(drive_url, local_path):
    response = requests.get(drive_url, allow_redirects=True)
    with open(local_path, "wb") as f:
        f.write(response.content)

# Direct download URLs
pkl_drive_url = "https://drive.google.com/uc?export=download&id=1VmmiFlmRWHaHE0SFgM13s14mc9qglZwq"
zip_drive_url = "https://drive.google.com/uc?export=download&id=1RU79RNR4cMiUuzsNZDcILpe0MG0g3c8Z"

# Mining_Documents.pkl
if not os.path.exists("Mining_Documents.pkl"):
    download_from_drive(pkl_drive_url, "Mining_Documents.pkl")
 

import gdown

url = "https://drive.google.com/uc?export=download&id=1RU79RNR4cMiUuzsNZDcILpe0MG0g3c8Z"
gdown.download(url, "new_faiss_index.zip", quiet=False)


if not os.path.exists("new_faiss_index"):
    with zipfile.ZipFile("new_faiss_index.zip", "r") as zip_ref:
        zip_ref.extractall()


st.set_page_config(page_title="MineVidya — Mining Q&A", layout="wide", initial_sidebar_state="expanded")

load_dotenv()

 
 
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('HF_TOKEN')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


llm=ChatGroq(model='groq/compound')

# prompt
prompt=ChatPromptTemplate.from_template(
    """
    You are MineVidya — an expert friendly AI assistant trained on Indian mining regulations, DGMS circulars, safety guidelines, and historical accident data. Your purpose is to assist mine engineers, safety officers, and regulatory managers in making safe, legal, and informed operational decisions.

 Role 1: Mining Safety Advisor  
When asked about safety, hazard response, or prevention:
- Reference relevant Indian mining laws (e.g., Mines Act 1952, CMR 2017, OMR 2017)
- Cite safety circulars or best practices from global standards where relevant
- If applicable, include historical accident learnings using structured format

 Role 2: Accident Analyst  
For queries about past accidents or incident types:
Respond using this format:
- **Incident Summary**: [What happened and where, e.g., "Cable fire at Jharia mine, 2021"]
- **Root Causes**: [Technical and human errors from reports]
- **Preventive Measures**: [What could have stopped it; link to DGMS circulars]
- **Regulatory Reference**: [e.g., Regulation 122, CMR 2017 or Mines Act Section 23]

 Role 3: Legal & Operations Expert  
For general regulatory or operational questions:
- Keep answers under 3 sentences
- Use concise, professional language
- Rely on current laws, technical knowledge, and document context

 Response Logic:
- Prioritize context from {context}  
- If context is insufficient but the question is clearly mining-related, reason using your internal mining knowledge.
- Do NOT hallucinate accident details — only refer if such cases exist in context or database metadata
 

Rules to Follow:
- If the question is related to mining but still the context does not contain relevant information, respond with: "False"
- If the question is unrelated to mining industry and services, respond with: "I'm here to assist only with mining-related queries, Feel free to ask mining related questions."
- If the question is related to mining but the question is not well defined, respond with: "Could you please clarify your mining-related question?"
- If the question is related to mining and the context contains relevant information, provide a detailed and accurate answer based on the context.

 Output Style:
- Respond in clear, formal tone, be polite.
- Use **bold headings** only when summarizing accident responses
- Avoid speculation; always link insight to documents, reports, or known mining practice
 
    <Question>
    {input}
    </Question>
"""
)

wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=250)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)

arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=250)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)

duck=DuckDuckGoSearchRun()

tools=[
    Tool(
        name="Wikipedia Search",
        func=wiki.run,
        description="Useful for when you need to look up general information on Wikipedia about mining or minerals."
    ),
    Tool(
        name="Arxiv Search",
        func=arxiv.run,
        description="Useful for when you need to look up research papers on Arxiv about mining or minerals."
    ),
    Tool(
        name="DuckDuckGo Search",
        func=duck.run,
        description="Useful for when you need to look up current news or web results about mining or minerals."
    )
]


template = """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought: {agent_scratchpad}"""

agent= create_react_agent(llm,tools,prompt=ChatPromptTemplate.from_template(template))
agent_executor=AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verborse=True
)

 

def create_vector_embedding():
    if 'retriever' not in st.session_state:
         
        with st.spinner("Loading existing vector database..."):
         
            st.session_state.embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2' )
            st.session_state.faiss = FAISS.load_local("new_faiss_index", st.session_state.embeddings,allow_dangerous_deserialization=True)
            with open('./Mining_Documents.pkl','rb') as f:
                st.session_state.documents = pickle.load(f)
            st.session_state.vectors=st.session_state.faiss.as_retriever(search_kwargs={"k":3})
            st.session_state.sparceRetriever=BM25Retriever.from_documents(st.session_state.documents)
            st.session_state.sparceRetriever.k=3
            st.session_state.retriever=EnsembleRetriever(retrievers=[st.session_state.vectors,st.session_state.sparceRetriever],
                                weights=[0.6,0.4])


def rag_retrieval(query: str):
    if not st.session_state.get('retriever'):
        return { 'answer': 'False' }, False
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(st.session_state.retriever, document_chain)
    response = retrieval_chain.invoke({'input': query})
    answer = response.get('answer', '')
    if "False" not in answer:
        return response, True
    return response, False

# ------------------------ Materials listing ----------------------------

def list_materials():
    """Return a curated list of MAIN reference materials used for the RAG system."""

    mining_resources = {
    "Legal and Regulatory Documents": [
        "The Mines Act, 1952",
        "The Mines Rules, 1955",
        "Coal Mines Regulations (CMR), 2017",
        "Mineral Conservation and Development Rules (MCDR), 2017",
        "Metalliferous Mines Regulations (MMR), 1961",
        "Mineral Concession Rules (MCR), 2016",
        "Offshore Minerals Concession and Development Rules (OMR), 2002 & 2024 amendments",
        "Oil Mines Regulations (OMR), 2017 & 1984 versions",
        "Central Electricity Authority Safety Regulations, 2010",
        "Government of India Gazette Notifications"
    ],
    "DGMS Circulars and Safety Alerts": [
        "DGMS Annual Reports (2005-2014 and ongoing)",
        "Safety Circulars: roof support design, pillar strength, machinery guarding, ventilation systems, dust control, blasting safety, electrical safety, worker training, accident investigation procedures, etc.",
        "Facility Alerts issued in response to specific mine incidents"
    ],
    "Accident Investigation Reports and Case Studies": [
        "Vindya UG Mine (2016)",
        "Pali UG Mine (2016)",
        "Codli Iron Ore Mine (2017)",
        "Amalgamated Konar Khasmahal OCP (2017)",
        "Kalne Iron Mine (2019)",
        "Numerous other undergrounds, open-pit, and specialized mines reports",
        "Coal company incidents: CCL, ECL, MCL, WCL (2017–2020)",
        "Specialized mining: ONGC drilling, granite, marble, chromite, manganese, copper, quartz/feldspar mines",
        "Incident categories: Fatal accidents, non-fatal injuries, near-misses, equipment failures, environmental incidents, safety violations",
        "Investigation findings: Root cause analysis, contributing factors, preventive measures, regulatory compliance status for each incident"
    ],
    "Technical Reference Materials": [
        "Duggal's Surveying Volumes 1 & 2 (2013)",
        "Engineering Surveying Reference Guide",
        "Global Mining Best Practices Documentation"
    ],
    "Supplementary Materials": [
        "Mine Safety and Health Administration (MSHA) guidelines adapted for Indian context",
        "International Labour Organization (ILO) conventions on mining safety and worker rights",
        "Environmental impact assessments for various mine types and scales",
        "Geological and hydrogeological reference materials for understanding mining site conditions"
    ]
}

    return [item for sublist in mining_resources.values() for item in sublist]

# ---------------------------- UI --------------------------------------

# Sidebar
with st.sidebar:
     
    st.header('MineVidya')
    st.markdown('**AI assistant for Indian mining regulations & safety**')
    page = st.radio('Navigation', ['Home', 'Chat Assistant', 'Materials', 'Settings'])
    st.markdown('---')
    st.caption('Built for mine engineers, safety officers, managers and exam aspirants.')

# Page header
st.markdown('<div class="title">MineVidya — Mining Q&A Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Ask about DGMS circulars, CMR/OMR regulation, historical accident learnings and safety best-practices.</div>', unsafe_allow_html=True)
st.markdown('---')

# Ensure model state loaded
with st.spinner('Initializing vector index and documents...'):
    create_vector_embedding()

# HOME
if page == 'Home':
    st.subheader('What this chatbot does')
    st.markdown(
        """
        - **MineVidya** is a retrieval-augmented chatbot built to answer mining-specific questions using a curated collection of Indian mining Acts, DGMS circulars, accident reports, and safety alerts.
        - It prefers evidence from all mining documents and only uses web search when the internal documents are insufficient.
        - Use it for regulatory lookups, accident analysis, preventive measures, training scenarios and exam preparation.
        """
    )

    st.info('Tip: Ask specific questions like "What does CMR 104 require?" or "Show similar accidents to X incident."')

    st.markdown('### Materials used for the RAG knowledge base')
    if st.button('Show materials list'):
        mats = list_materials()
        if mats:
            st.success(f'{len(mats)} materials found')
            for m in mats:
                st.write('- ', m)
        else:
            st.warning('No materials found. Ensure Mining_Documents.pkl or ./documents folder exists.')

    st.markdown('### Quick actions')
    col1, col2 = st.columns([1,1])
     
    with col1:
        if st.button('Reload Index'):
            # force reload
            if 'retriever' in st.session_state:
                del st.session_state['retriever']
            create_vector_embedding()
            st.success('Index reloaded')
    with col2:
        if st.button('Show sample question'):
            st.info('Try: "List preventive measures for conveyor belt fire incidents."')

# MATERIALS page (standalone)
elif page == 'Materials':
    st.subheader('Materials & Documents')
    mats = list_materials()
    if not mats:
        st.warning('No materials found. Ensure Mining_Documents.pkl or ./documents exist.')
    else:
        st.write(f'Found **{len(mats)}** materials:')
        for i,m in enumerate(mats, start=1):
            with st.expander(f'{i}. {os.path.basename(m)}'):
                st.write(m)
                 
                docs = st.session_state.get('documents', [])
                for d in docs:
                    src = d.metadata.get('source') if hasattr(d, 'metadata') else None
                    if src and os.path.basename(src) == os.path.basename(m):
                        st.write('\nPreview:')
                        st.write((d.page_content[:800] + '...') if len(d.page_content) > 800 else d.page_content)
                        break

# CHAT ASSISTANT
elif page == 'Chat Assistant':
    st.subheader('Ask MineVidya')

    # chat area
    chat_container = st.container()
    input_col, send_col = st.columns([7,1])

    with input_col:
        user_query = st.text_area('Your question', height=110, key='user_query')
    with send_col:
        send = st.button('Send')

    if send and user_query:
        with st.spinner('Searching documents...'):
            response, found = rag_retrieval(user_query)
        if found:
            # show answer and context
            st.write(response.get('answer'))
            with st.expander('Context Documents (hybrid search)'):
                for doc in response.get('context', []):
                    st.write(doc.page_content)
                    st.write('---')
        else:
            st.warning('Internal docs insufficient — falling back to web agent')
            with st.spinner('Using external search agent...'):
                agent_response = agent_executor.invoke({"input": user_query})
                st.write(agent_response.get('output'))

# SETTINGS
elif page == 'Settings':
    st.subheader('Settings & Diagnostics')
    st.write('Model: llama-3.1-8b-instant (ChatGroq)')
    st.write('Embeddings: sentence-transformers/all-MiniLM-L6-v2')
    st.write('FAISS index folder exists: ', os.path.exists('new_faiss_index'))
    st.write('Documents pickle exists: ', os.path.exists('Mining_Documents.pkl'))
    if st.button('Clear session state'):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.experimental_rerun()

# Footer
st.markdown('---')
st.caption('MineVidya | RAG-powered mining assistant — built for internal use.')





