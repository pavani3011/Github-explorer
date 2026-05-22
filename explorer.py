import os
import base64
import requests

from langchain_community.document_loaders import GithubFileLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_core.prompts import(
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from pinecone import Pinecone, ServerlessSpec

from config import (GITHUB_TOKEN, PINECONE_API, OPENAI_API_KEY,
                    REPO, BRANCH, INDEX_NAME,FILE_EXTENSIONS,
                    CHUNK_OVERLAP, CHUNK_SIZE, TOP_K)

def load_github_docs(
        repo: str,
        token: str,
        branch: str,
        extensions: tuple,
) -> list[Document]:
    try:
        loader = GithubFileLoader(
            repo=repo,
            branch=branch,
            access_token= token,
            github_api_url="https://api.github.com",
            file_filter=lambda path: path.endswith(extensions),
        )
        docs = loader.load()
        print(f"GithubFileLoader: {len(docs)} files loaded")
        return docs
    except Exception as exc:
        print(f"Github failed ({exc}), falling back to Trees Api...")
        return _load_via_trees_api(repo, token, branch, extensions)
    

def _load_via_trees_api(
        repo:str,
        token: str,
        branch:str,
        extensions: tuple,
)-> list[Document]:
    headers = {
        "Authorization":f"Bearer {token}",
        "Accept":"application/vnd.github+json",
    }
    base = "https://api.github.com"
    
    tree = requests.get(
        f"{base}/repos/{repo}/git/trees/{branch}?recursive=1",
        headers=headers,
        timeout=30,
    ).json()

    blobs = [
        item for item in tree.get("tree", [])
        if item["type"]== "blob" and item["path"].endswith(extensions)
    ]
    print(f"Trees API: {len(blobs)} matching files found.")

    docs= []
    for item in blobs:
        path = item["path"]
        resp= requests.get(
            f"{base}/repos/{repo}/contents/{path}?ref={branch}",
            headers=headers,
            timeout=30,
        ).json()

        raw = base64.b64decode(resp["content"]).decode("utf-8", errors="ignore")
        docs.append(Document(
            page_content=raw,
            metadata={
                "path":path,
                "sha": item["sha"],
                "source":f"https://github.com/{repo}/blob/{branch}/{path}",
            },
        ))
    return docs

def split_and_enrich(
        raw_docs: list[Document],
        repo:str,
        branch:str,
        chunk_size: int,
        chunk_overlap: int,
)-> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= chunk_size,
        chunk_overlap= chunk_overlap,
        add_start_index = True,
        separators=["\n## ", "\n### ", "\n\n", "\n", " ", ""],
    )
    docs = splitter.split_documents(raw_docs)

    for doc in docs:
        path = doc.metadata.get("path", "")
        doc.metadata["source"] = f"https://github.com/{repo}/blob/{branch}/{path}"
        doc.metadata["repo"] = repo
        doc.metadata["branch"] = branch
        doc.metadata["file_path"] = path
        doc.metadata["file_type"] = path.rsplit(".",1)[-1] if "." in path else "unknown"

    print(f" split into {len(docs)} chunks")
    return docs
