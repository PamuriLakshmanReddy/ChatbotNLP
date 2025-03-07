from src.helper import load_pdf_file,text_split,download_hugging_face_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv, find_dotenv
import os
load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"]=PINECONE_API_KEY

extracted_data = load_pdf_file(data="E:/ENd to end proects/ChatbotNLP/Data/")
text_chunk=text_split(extracted_data)
embeddings_model=download_hugging_face_embeddings()

PINECONE_API_KEY=os.environ.get("PINECONE_API_KEY")
pc=Pinecone(api_key=PINECONE_API_KEY)
index_name="nlpchatbot"
pc.create_index(name=index_name,dimension=384,metric="cosine",spec=ServerlessSpec(cloud="aws",region="us-east-1"))
docsearch = PineconeVectorStore.from_documents(text_chunk,embeddings_model,index_name=index_name)
