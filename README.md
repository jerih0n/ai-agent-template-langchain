# LangChain Chat Agent Template

> **A fast-start template for prototyping and MVPs.**  
> Clone it, drop in your system prompt and tools, and you have a working
> chat agent with persistent memory in minutes — not days.  
> OpenAI is the default, but you are **not locked in**: swap to Anthropic,
> Google, Mistral, or any other provider supported by LangChain with a
> single line change.

A ready-to-run Python template for building a chat agent powered by the [LangChain `create_agent` API](https://docs.langchain.com/oss/python/langchain/overview).

| Feature | Technology |
|---|---|
| Chat UI with thread management | [Gradio Blocks](https://www.gradio.app/docs/gradio/blocks) |
| Agent loop (tool calling, memory) | [LangChain `create_agent`](https://docs.langchain.com/oss/python/langchain/agents) |
| Persistent conversation memory | [LangGraph PostgresSaver checkpointer](https://docs.langchain.com/oss/python/langgraph/overview) |
| Auto thread summarisation | [`@after_agent` middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom) |
| Config & validation | [`pydantic-settings`](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| Database migrations | Custom asyncpg runner (`app/database/sql_database/migrations_manager.py`) |

---

## Prerequisites

| Requirement | Minimum version |
|---|---|
| Python | 3.11+ |
| PostgreSQL | 14+ (local or cloud) |
| OpenAI API key | Any active key |

---

## Setup

### 1 — Clone / copy the project

```bash
# Clone or download the repo
cd ai-agent-template-langchain
```

### 2 — Start PostgreSQL

PostgreSQL is **required** for this template.

If you want the fastest local setup, use Docker:

```bash
docker compose up -d postgres
```

This starts Postgres on `127.0.0.1:5432` with a default database named `chat_agent`.

### 3 — Enter the app directory

```bash
cd src
```

### 4 — Create a virtual environment

**Windows (PowerShell)**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 5 — Install dependencies

```bash
pip install -r requirements.txt
```

### 6 — Configure environment variables

Copy the example file and fill in your values:

**Windows**
```powershell
copy env.example .env
```

**macOS / Linux**
```bash
cp env.example .env
```

Then open `.env` in your editor and set at minimum:

```dotenv
OPENAI_API_KEY=sk-...
POSTGRES_URL=postgresql://user:password@localhost:5432/chat_agent
```

> **Note on the Postgres URL format**
>
> Use the standard `postgresql://` scheme.
> The template also accepts `postgresql+psycopg://` and normalises it internally.

**Full list of variables** (all optional except the two above):

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | — | **Required.** Your OpenAI key. |
| `POSTGRES_URL` | — | **Required.** PostgreSQL connection string for threads + LangGraph memory. |
| `DATABASE_URL` | — | Heroku-style alias for `POSTGRES_URL`. |
| `APP_NAME` | `AI Chat Agent` | Title shown in the browser tab and UI heading. |
| `APP_HOST` | `127.0.0.1` | Host the Gradio server binds to. |
| `APP_PORT` | `7860` | Port the Gradio server listens on. |
| `AGENT_MODEL` | `gpt-4o-mini` | Any LangChain-supported model string. |
| `AGENT_TEMPERATURE` | `0.2` | LLM temperature (0.0 – 2.0). |

### 7 — Run the app

```bash
python run.py
```

The Gradio interface opens at **http://127.0.0.1:7860**.

On first run, the template will automatically:
1. Create the `chat_agent` database if it does not exist.
2. Run any pending SQL migrations (creates the `threads` table).
3. Set up the LangGraph checkpointer tables (`checkpoints`, `checkpoint_writes`, etc.).

The app starts in a fresh state with no thread selected by default. You can create a thread with `New Conversation`, or just send the first message and let the app create one automatically.

---

## Project structure

```
.
├── docker-compose.yml                   # Optional local Postgres for the fastest setup
├── requirements-dev.txt                 # Smoke-test dependencies
├── tests/
│   ├── test_startup.py
│   └── test_ui.py
└── src/
    ├── env.example                      # Copy to .env and fill in your values
    ├── run.py                           # Entry point — migrations → agent → Gradio
    ├── requirements.txt
    │
    └── app/
        ├── core/
        │   └── config.py                    # pydantic-settings config (reads .env)
        │
        ├── ai/
        │   ├── agent.py                     # ChatAgent — wraps create_agent()
        │   ├── prompts/
        │   │   ├── default_system_prompt.md # ← CUSTOMISE: your agent's persona
        │   │   └── thread_summary_prompt.md # Prompt used by the auto-summary middleware
        │   ├── middlewares/
        │   │   └── thread_summary.py        # @after_agent hook — labels new threads
        │   └── memory/
        │       └── short_lived_memory/
        │           └── short_lived_memory_manager.py  # PostgresSaver + thread CRUD
        │
        ├── database/
        │   └── sql_database/
        │       ├── postgres_db_helper.py    # asyncpg (async) + psycopg3 (sync) helpers
        │       ├── migrations_manager.py    # Discovers and runs *.sql migration files
        │       ├── migrations/
        │       │   └── 001_create_threads_table.sql
        │       └── commands/
        │           └── threads_commands.py  # Thread CRUD (INSERT / SELECT / DELETE)
        │
        ├── tools/
        │   └── example_tool.py              # Example LangChain tool to replace or extend
        │
        └── ui/
            └── ui.py                        # Gradio Blocks chat interface
```

---

## Customize First

If you are using this as a starter template, these are the first files to edit:

| File | Why you will likely change it |
|---|---|
| `app/ai/prompts/default_system_prompt.md` | Define your agent's role, tone, and rules |
| `app/tools/example_tool.py` | Replace the example tool with your own tools |
| `app/ai/agent.py` | Register tools, model config, and middleware |

---

## Customisation

### Change the agent's persona

Edit [`app/ai/prompts/default_system_prompt.md`](app/ai/prompts/default_system_prompt.md).  
Replace the placeholder text with your agent's role, tone, and domain knowledge.

### Add tools to the agent

Start by editing [`app/tools/example_tool.py`](app/tools/example_tool.py), then open [`app/ai/agent.py`](app/ai/agent.py) and add any callable or
[LangChain tool](https://docs.langchain.com/oss/python/langchain/tools) to the `tools=[]` list:

```python
from my_tools import search_web, run_code

return create_agent(
    model=self._llm,
    tools=[search_web, run_code],   # ← add your tools here
    system_prompt=self.system_prompt,
    checkpointer=self._checkpointer,
    middleware=[summarise_if_new],
)
```

See the [Tools docs](https://docs.langchain.com/oss/python/langchain/tools) for how to write and register tools.

### Swap the LLM provider (you are not locked in to OpenAI)

Replace the two lines in [`app/ai/agent.py`](app/ai/agent.py) that create
`self._llm` with any [LangChain-supported model](https://docs.langchain.com/oss/python/integrations/providers/overview).
Install the matching package, update the import, and the rest of the template
works without any other change.

```python
# Anthropic — pip install langchain-anthropic
from langchain_anthropic import ChatAnthropic
self._llm = ChatAnthropic(model="claude-sonnet-4-6", api_key="...")

# Google Gemini — pip install langchain-google-genai
from langchain_google_genai import ChatGoogleGenerativeAI
self._llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", google_api_key="...")

# Mistral — pip install langchain-mistralai
from langchain_mistralai import ChatMistralAI
self._llm = ChatMistralAI(model="mistral-large-latest", api_key="...")

# Ollama (local, no API key needed) — pip install langchain-ollama
from langchain_ollama import ChatOllama
self._llm = ChatOllama(model="llama3.2")

# Azure OpenAI — pip install langchain-openai
from langchain_openai import AzureChatOpenAI
self._llm = AzureChatOpenAI(azure_deployment="gpt-4o", ...)
```

> See the full list of providers at
> https://docs.langchain.com/oss/python/integrations/providers/overview

### Add a new database migration

Create a new numbered `.sql` file in `app/database/sql_database/migrations/`:

```
002_add_tags_table.sql
```

It will be picked up and executed automatically on the next `python run.py`.

### Add a custom `@after_agent` middleware hook

Follow the pattern in [`app/ai/middlewares/thread_summary.py`](app/ai/middlewares/thread_summary.py)
and register the hook in `_build_agent()`:

```python
middleware=[summarise_if_new, your_custom_hook]
```

See the [Middleware docs](https://docs.langchain.com/oss/python/langchain/middleware/custom) for the full API.

---

## Running tests

Minimal smoke tests are included for startup validation and the default UI thread flow.

Install dev dependencies from the repository root:

```bash
pip install -r requirements-dev.txt
pytest
```

---

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `OPENAI_API_KEY` error on startup | Key missing from `.env` | Add `OPENAI_API_KEY=sk-...` to `.env` |
| `Missing required environment variables` | `.env` is missing required values | Copy `env.example` to `.env` and fill in `OPENAI_API_KEY` and `POSTGRES_URL` |
| `POSTGRES_URL is required` | URL missing from `.env` | Add `POSTGRES_URL=postgresql://...` to `.env` |
| `can't subtract offset-naive and offset-aware datetimes` | Old code path (fixed) | Pull latest and re-run |
| Thread list always empty | DB not reachable at startup | Check Postgres is running and the URL is correct |
| Gradio `ValueError: list of choices` warnings | Harmless Gradio 6.x warning | Ignore — does not affect functionality |

---

## Key dependencies

| Package | Purpose | Docs |
|---|---|---|
| `langchain` | `create_agent`, middleware, tool calling | [docs](https://docs.langchain.com/oss/python/langchain/overview) |
| `langchain-openai` | `ChatOpenAI` model wrapper | [docs](https://docs.langchain.com/oss/python/integrations/providers/openai) |
| `langgraph` | Agent graph execution and checkpointing | [docs](https://docs.langchain.com/oss/python/langgraph/overview) |
| `langgraph-checkpoint-postgres` | `PostgresSaver` — persistent memory | [PyPI](https://pypi.org/project/langgraph-checkpoint-postgres/) |
| `psycopg[binary]` | Sync psycopg3 driver (used by checkpointer) | [docs](https://www.psycopg.org/psycopg3/docs/) |
| `asyncpg` | Async Postgres driver (used for thread CRUD) | [docs](https://magicstack.github.io/asyncpg/current/) |
| `gradio` | Chat UI | [docs](https://www.gradio.app/docs) |
| `pydantic-settings` | Env-var config with validation | [docs](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) |
| `python-dotenv` | Loads `.env` into `os.environ` | [PyPI](https://pypi.org/project/python-dotenv/) |
