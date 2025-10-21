import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
load_dotenv()
import pickle

os.environ['HF_TOKEN']=os.getenv('HF_TOKEN')

with open("Mining_Documents.pkl", "rb") as f:
    documents = pickle.load(f)
 
embeddings=HuggingFaceEmbeddings(model='sentence-transformers/all-MiniLM-L6-v2')

faiss=FAISS.from_documents(documents=documents,embedding=embeddings)

faiss.save_local("new_faiss_index")
