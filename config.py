import os
from dotenv import load_dotenv

load_dotenv()

GITHUB_TOKEN = os.environ["github_token"]
OPENAI_API_KEY = os.environ["openai_api"]
PINECONE_API= os.environ["pinecone_api"]


REPO = "langchain-ai/langchain"
BRANCH= "master"
INDEX_NAME= "github-docs"
FILE_EXTENSIONS= (".md", ".py", ".rst")

CHUNK_SIZE= 1000
CHUNK_OVERLAP= 150
TOP_K= 5

