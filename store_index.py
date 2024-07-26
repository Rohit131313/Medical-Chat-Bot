from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from pinecone import Pinecone
import pinecone
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API = os.getenv("PINECONE_API_KEY")


extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API)


index_name="medical-bot"

#Creating Embeddings for Each of The Text Chunks & storing
vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)

vectorstore.add_texts([t.page_content for t in text_chunks])