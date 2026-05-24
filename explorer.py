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


def build_vectorstore(
        docs: list[Document],
        pinecone_api_key:str,
        openai_api_key :str,
        index_name: str,
)->PineconeVectorStore:
    print(f"\n Upserting {len(docs)} chunks into pinecone index '{index_name}'...")

    pc = Pinecone(api_key=pinecone_api_key)
    existing = [i.name for i in pc.list_indexes()]
    if index_name not in existing:
        pc.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec = ServerlessSpec(cloud="aws", region="us-east-1"),
        )
        print(f" created new index : {index_name}")
    else:
        print(f" reusing existing index: {index_name}")

    embeddings = OpenAIEmbeddings(
        model= "text-embedding-3-small",
        openai_api_key = openai_api_key,
    )

    vectorstore = PineconeVectorStore.from_documents(
        documents=docs,
        embedding= embeddings,
        index_name= index_name,

    )

    print("upsert complete.")
    return vectorstore

def build_prompt(repo:str)-> ChatPromptTemplate:
    examples = [
        {
            "question": "What does the 'BaseLLM.__call__' method do?",
            "answer": (
                "`BaseLLM.__call__` is the entry point when you invoke a LangChain LLM "
                "like a function. Internally it delegates to `generate()`, which handles "
                "batching and returns an `LLMResult`. It also fires callbacks "
                "(on_llm_start, on_llm_end) so the tracing layer can record inputs and "
                "outputs. Key point: it always wraps the raw string in an "
                "`LLMResult.generations` list, even for single prompts."
            ),
        },
        {
            "question": "How does `RecursiveCharacterTextSplitter` decide where to split?",
            "answer": (
                "It tries each separator in order — by default "
                r'["\n\n", "\n", " ", ""]'
                " — and splits on the first one that keeps chunks ≤ chunk_size. "
                "This preserves paragraph → sentence → word structure. "
                "When chunk_overlap > 0, the tail of one chunk is prepended to "
                "the next, so context isn't lost across boundaries."
            ),
        },
        {
            "question": "What is the purpose of `ConversationBufferMemory`?",
            "answer": (
                "`ConversationBufferMemory` stores every human/AI message pair as plain "
                "text in a growing buffer. On each new turn it injects the full history "
                "into the prompt under the `history` key. Unlike "
                "`ConversationSummaryMemory` it does not compress, so it preserves exact "
                "wording at the cost of growing token usage per turn."
            ),
        },
    ]
    
    example_prompt = ChatPromptTemplate([
        ("human", "{question}"),
        ("ai","{answer}"),
    ])

    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            "You are a senior software engineer explaining code from the "
            f"{repo} GitHub repository.\n"
            "Rules:\n"
            "  • Always cite the source file when referencing code.\n"
            "  • Prioritise the retrieved context; use general knowledge only as a last resort.\n"
            "  • Be concise, precise, and highlight edge cases.\n\n"
            "Context from the repository:\n{context}"
        ),
        few_shot,
        HumanMessagePromptTemplate.from_template("{question}"),
    ])
    return prompt

def build_chain(
        vectorstore: PineconeVectorStore,
        prompt:ChatPromptTemplate,
        openai_api_key:str,
        top_k: int,
)-> ConversationalRetrievalChain:
    print("\nBuilding ConversationalRetrievalChain...")

    retriever = vectorstore.as_retriever(
        search_Type = "similarity",
        search_kwargs= {"k": top_k},
    )

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )

    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0,
        openai_api_key=openai_api_key,
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt":prompt},
        return_source_documents=True,
        verbose=False,
    )

    print("Chain ready.\n")
    return chain

