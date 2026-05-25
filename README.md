# GitHub Documentation Assistant 🤖

A RAG (Retrieval-Augmented Generation) pipeline that lets you query any GitHub repository's documentation using natural language. Built with LangChain, Pinecone, and OpenAI.

---

## How It Works

```
GitHub Repo  →  Loader  →  Chunker  →  Pinecone (vectors)
                                              ↓
User Question  →  Retriever  →  Few-Shot Prompt  →  GPT-4o  →  Answer + Sources
```

1. **Loads** files from any public or private GitHub repo
2. **Embeds** and upserts chunks into Pinecone with metadata (source URL, file path)
3. **Prompts** the LLM using a Few-Shot template for consistent code explanations
4. **Answers** questions via `ConversationalRetrievalChain` with full chat memory

---

## Project Structure

```
GITHUB-DOC-ASSISTANT/
├── .env              # API keys (never commit this)
├── .gitignore        # excludes .env and __pycache__
├── config.py         # all constants (REPO, CHUNK_SIZE, TOP_K, etc.)
├── explorer.py       # query logic and ask() helper
├── main.py           # entry point — orchestrates all 4 steps
└── README.md
```

---

## Setup

### 1. Clone the repo
```bash
git clone https://github.com/your-username/github-doc-assistant.git
cd github-doc-assistant
```

### 2. Install dependencies
```bash
pip install langchain langchain-community langchain-openai \
            langchain-pinecone pinecone-client openai tiktoken
```

### 3. Create your `.env` file
```bash
GITHUB_TOKEN=ghp_your_token_here
OPENAI_API_KEY=sk-your_key_here
PINECONE_API_KEY=your_pinecone_key_here
```

> **Where to get the keys:**
> - `GITHUB_TOKEN` → [GitHub Settings → Developer Settings → Personal Access Tokens](https://github.com/settings/tokens) (need `repo:read` scope)
> - `OPENAI_API_KEY` → [platform.openai.com](https://platform.openai.com/api-keys)
> - `PINECONE_API_KEY` → [app.pinecone.io](https://app.pinecone.io)

### 4. Configure the target repo
Edit `config.py`:
```python
REPO    = "facebook/react"      # any public repo, or your own private repo
BRANCH  = "main"
```

### 5. Run
```bash
python main.py
```

---

## Usage

```
============================================================
GitHub Documentation Assistant — ready!
Repo: langchain-ai/langchain   |   Type 'exit' to quit
============================================================

You: How does the AgentExecutor handle tool errors?

📖  Answer:
The AgentExecutor catches ToolException and routes it through
handle_tool_error, which can be a bool, string, or callable...

🔗  Sources:
    • https://github.com/langchain-ai/langchain/blob/master/libs/langchain/langchain/agents/agent.py

You: And how does that relate to max_iterations?
...
```

---

## Configuration Reference

All settings live in `config.py`:

| Variable | Default | Description |
|---|---|---|
| `REPO` | `langchain-ai/langchain` | Target GitHub repo (`owner/repo`) |
| `BRANCH` | `master` | Branch to load from |
| `INDEX_NAME` | `github-docs` | Pinecone index name |
| `FILE_EXTENSIONS` | `(.md, .py, .rst)` | File types to load |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `150` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |

---

## Supported Repos

| Type | Supported | Requirement |
|---|---|---|
| Any public repo | ✅ | Valid GitHub token |
| Your own private repo | ✅ | Token with `repo` scope |
| Your org's private repo | ✅ | Token with `repo` scope + org access |
| Someone else's private repo | ❌ | Must be a collaborator |

---

## Tech Stack

| Tool | Purpose |
|---|---|
| [LangChain](https://github.com/langchain-ai/langchain) | Document loading, chaining, memory |
| [Pinecone](https://www.pinecone.io/) | Vector database for embeddings |
| [OpenAI](https://openai.com/) | Embeddings (`text-embedding-3-small`) + LLM (`gpt-4o`) |
| [GitHub API](https://docs.github.com/en/rest) | Fetching repo files |

---

## .gitignore

Make sure your `.gitignore` contains:
```
.env
__pycache__/
*.pyc
*.pyo
.DS_Store
```

---

## License

MIT License — use freely, modify as needed.