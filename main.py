from explorer import (
    load_github_docs,_load_via_trees_api,split_and_enrich,
    build_vectorstore,build_prompt,build_chain, ask)
from config import (GITHUB_TOKEN, PINECONE_API, OPENAI_API_KEY,
                    REPO, BRANCH, INDEX_NAME,FILE_EXTENSIONS,
                    CHUNK_OVERLAP, CHUNK_SIZE, TOP_K)

def main() -> None:
    raw_docs = load_github_docs(REPO,GITHUB_TOKEN,BRANCH, FILE_EXTENSIONS)
    docs = split_and_enrich(raw_docs,REPO,BRANCH,CHUNK_SIZE,CHUNK_OVERLAP)

    vectorstore = build_vectorstore(docs, PINECONE_API, OPENAI_API_KEY, INDEX_NAME)

    prompt = build_prompt(REPO)

    chain = build_chain(vectorstore,prompt, OPENAI_API_KEY, TOP_K)

    print("="*60)
    print("Github documentation assistant - ready!")
    print(f"Repo: {REPO} | type 'exit' to quit")
    print("="*60)

    while True:
        try:
            question = input("\nYou: ").strip()
        except (EOFError,KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit","quit", "q"}:
            print("Goodbye!")
            break
        
        ask(chain,question)


if __name__ == "__main__":
    main()