from langchain_community.tools import DuckDuckGoSearchRun
from pinecone_text.sparse import BM25Encoder
from langchain_community.retrievers import PineconeHybridSearchRetriever
from pinecone import Pinecone
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os
import warnings
import requests
from PIL import Image
from io import BytesIO

warnings.filterwarnings('ignore')
load_dotenv()
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("assistant")
HF_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
bm25_encoder = BM25Encoder().default()
retriever = PineconeHybridSearchRetriever(embeddings=embeddings, sparse_encoder=bm25_encoder ,index=index)


# Define a very simple tool function that returns the current time
def get_current_time(*args, **kwargs):
    """Returns the current time in H:MM AM/PM format."""
    import datetime  # Import datetime module to get current time

    now = datetime.datetime.now()  # Get current time
    time = now.strftime("%I:%M %p")  # Format time in H:MM AM/PM format
    return f"The current time is {time}"

def update_db(text):
    if type(text) != str:
        return f"Please pass a single string, eg. \'Text\', the current type of input passed is {type(text)}"
    retriever.add_texts([text])
    print("Memory Updated:", text)
    return 'Texts added'

def retrieve(text):
    if type(text) != str:
        return 'please pass a single string, database not updated'
    l = []
    ret = retriever.invoke(text)
    for d in ret:
        l.append(d.page_content)
    return l

def web_search(query):
    search = DuckDuckGoSearchRun()
    return search.run(query)

def end_conversation(blank):
    print()
    print("Exiting Conversation..")
    exit()

def generate_image(text):
    image_url = f"https://image.pollinations.ai/prompt/{text}"
    response = requests.get(image_url)
    image = Image.open(BytesIO(response.content))
    image.show()
    image.save("RA/image.jpg", "JPEG")