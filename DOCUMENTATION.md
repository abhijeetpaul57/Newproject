# Comprehensive Project & Architecture Documentation
## Self-Corrective RAG Technical Documentation Assistant

---

## 1. Executive Summary & Project Purpose

The **RAG Technical Documentation Assistant** is a production-grade, self-corrective Retrieval-Augmented Generation (RAG) system built for technical developer documentation.

### The Problem with Naive RAG
Standard (naive) RAG architectures follow a rigid, unidirectional pipeline:
```
Query ──► Vector Search ──► Top-K Chunks ──► LLM Generation
```
This naive approach exhibits critical failure modes in real-world technical workflows:
1. **Semantic Misalignment**: User questions often use colloquial phrasing, abbreviations, or missing keywords that do not match the exact terminology used in official documentation.
2. **Context Pollution (Noise)**: Vector similarity returns chunks that may share keywords but are contextually irrelevant or pure boilerplate, confusing the LLM.
3. **Hallucination on Retrieval Failure**: When retrieved context contains no answer, naive systems either hallucinate plausible-sounding falsehoods or produce misleading answers.
4. **Zero Self-Correction**: If the initial retrieval fails, naive systems cannot adjust their strategy or retry.

### The Self-Corrective Solution
This project solves these limitations by modeling retrieval and synthesis as a **directed cyclic state machine** using **LangGraph `StateGraph`**:
- **Query Rewriting & Classification**: Understands the user's intent and expands technical terminology.
- **Batch Document Relevance Grading**: An LLM evaluator grades each retrieved chunk as `relevant` or `irrelevant`.
- **Cyclic Feedback Loop**: If no relevant documents are found, the state machine loops back to reformulate the query with alternative synonyms and re-retrieves, up to a configurable retry limit (`max_retries = 2`).
- **Strict Grounded Generation**: When relevant documents exist, answers are synthesized strictly from the context with exact inline citations (`[Source N: file — heading]`).
- **Honest General Knowledge Fallback**: If retries are exhausted or off-topic questions are asked, the system explicitly alerts the user with a formatted notice block rather than fabricating citations.

---

## 2. High-Level System Architecture

The project consists of four interconnected tiers:
1. **Frontend Presentation Tier**: Interactive Streamlit Web UI featuring dark glassmorphism styling, real-time pipeline telemetry, corpus management, and feedback logging.
2. **REST API Service Tier**: FastAPI application providing async OpenAPI endpoints with Pydantic v2 schemas and validation.
3. **Workflow Orchestration Tier**: LangGraph cyclic state machine controlling execution flow, state transitions, and retry loops.
4. **Data & Storage Tier**: Persistent ChromaDB vector store paired with local CPU embeddings (`sentence-transformers/all-MiniLM-L6-v2`) and append-only JSONL feedback logging.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        Streamlit Frontend (ui.py)                     │
│  - Dark Glassmorphism UI (Outfit & Plus Jakarta Sans typography)       │
│  - Real-Time Pipeline Inspector & Source Citations                     │
│  - Document Ingestion (File Drag-and-Drop & URL Importer)              │
│  - Thumbs Up / Down Feedback Widget                                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP REST
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI Backend (app/main.py)                   │
│  - POST /query         - POST /ingest           - GET /health          │
│  - GET /documents      - POST /ingest/url       - POST /feedback       │
│  - DELETE /documents   - DELETE /documents/{id}                        │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   LangGraph Workflow Engine (rag/graph.py)              │
│                                                                        │
│                    [START] ──► query_analysis                          │
│                                     │                                  │
│             ┌───────────────────────┴───────────────────────┐          │
│             │ (greeting)                                    │ (normal) │
│             ▼                                               ▼          │
│         fallback ◄─── (exhausted/error) ◄─── grading ◄── retrieval     │
│             │                                   │                      │
│             │                           (has relevant)                 │
│             ▼                                   ▼                      │
│           [END] ◄─────────────────────────── generate                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                             ▼
       ┌───────────────────────────┐ ┌───────────────────────────┐
       │   ChromaDB Vector Store   │ │    Google Gemini LLM      │
       │ - Persistent Collection   │ │ - gemini-3.8-flash        │
       │ - all-MiniLM-L6-v2 (384d) │ │ - Structured Output       │
       │ - SHA-256 Deduplication   │ │ - Temperature Control     │
       └───────────────────────────┘ └───────────────────────────┘
```

---

## 3. LangGraph Workflow & State Machine Deep-Dive

### The RAG State Schema (`rag/state.py`)

All graph nodes read from and write to a shared, immutable-key `RAGState` TypedDict:

```python
class SourceMetadata(TypedDict):
    doc_id: str             # Document identifier (e.g., 'doc_fastapi_guide')
    source: str             # File name or URL (e.g., 'fastapi_guide.md')
    chunk_id: str           # Unique chunk ID (e.g., 'doc_fastapi_guide_chunk_0003')
    heading: str            # Nearest Markdown section header

class RAGState(TypedDict):
    original_question: str          # Original question from the user (immutable)
    top_k: int                      # Number of vector chunks to retrieve
    rewritten_query: str            # Query optimized for embedding search
    query_type: str                 # conceptual | how-to | troubleshooting | api-reference | greeting
    retrieved_documents: list[Doc]  # Raw chunks returned by ChromaDB
    relevant_documents: list[Doc]   # Chunks confirmed relevant by the grading node
    retry_count: int                # Current self-correction retry iteration
    max_retries: int                # Upper bound for retry loop (default: 2)
    answer: str                     # Synthesized answer or fallback statement
    sources: list[SourceMetadata]   # Structured citations
    generation_supported: bool      # True if answer is grounded in docs
    error: Optional[str]            # Propagated exception details, if any
```

### Detailed Node Execution Flow

```
                     ┌───────────────────────┐
                     │         START         │
                     └───────────┬───────────┘
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │    query_analysis     │
                     │  - classify intent    │
                     │  - expand terminology │
                     └───────────┬───────────┘
                                 │
                 [route_after_query_analysis]
                     /                       \
        query_type == 'greeting'          standard query
                   /                           \
                  ▼                             ▼
        ┌───────────────────┐         ┌───────────────────┐
        │     fallback      │         │     retrieval     │
        │ - friendly intro  │         │ - ChromaDB search │
        └─────────┬─────────┘         │ - top_k chunks    │
                  │                   └─────────┬─────────┘
                  │                             │
                  │                             ▼
                  │                   ┌───────────────────┐
                  │                   │      grading      │
                  │                   │ - batch LLM grade │
                  │                   │ - filter relevant │
                  │                   └─────────┬─────────┘
                  │                             │
                  │                   [route_after_grading]
                  │                       /        │        \
                  │           relevant docs?    no docs?   error/exhausted
                  │                 /              │              \
                  │                ▼               ▼               ▼
                  │       ┌──────────────┐  retry < max?     ┌──────────────┐
                  │       │   generate   │   /         \     │   fallback   │
                  │       │ - grounded   │  yes        no    │ - general AI │
                  │       │ - citations  │   │          └─►  │   knowledge  │
                  │       └──────┬───────┘   │               └──────┬───────┘
                  │              │     (loop back)                  │
                  │              │           ▼                      │
                  │              │  ┌─────────────────┐             │
                  │              │  │ query_analysis  │             │
                  │              │  └─────────────────┘             │
                  │              │                                  │
                  ▼              ▼                                  ▼
                [END]          [END]                              [END]
```

#### Node 1: `query_analysis` (`rag/nodes/query_analysis.py`)
1. **Greeting Fast-Path**: Normalizes text and inspects tokens. If the user input is a pure greeting (`"hi"`, `"hello"`, `"good morning"`), it routes directly to `fallback` without wasting vector lookups or grading calls.
2. **Classification**: Categorizes queries into one of five technical types:
   - `conceptual`: High-level theory, definitions, core ideas.
   - `how-to`: Concrete coding tasks, configuration steps, recipes.
   - `troubleshooting`: Bug fixes, exceptions, error resolution.
   - `api-reference`: Method arguments, return types, class signatures.
   - `conversational`: Informal or general queries.
3. **Semantic Query Rewriting**: Expands acronyms (`RAG` → `Retrieval-Augmented Generation`, `LCEL` → `LangChain Expression Language`), injects technical terminology, and removes conversational noise.
4. **Retry Awareness**: When running on retry iterations (`retry_count > 0`), the LLM receives the prior query that failed and produces alternative technical synonyms and broader search terms.

#### Node 2: `retrieval` (`rag/nodes/retrieval.py`)
1. Executes similarity search against the persistent ChromaDB collection using the `rewritten_query` for `top_k` results.
2. Preserves all metadata: `source` filename, `heading`, `doc_id`, `chunk_id`, and `chunk_index`.
3. Checks for exact document matches to ensure relevant chunks are pulled into the candidate set.

#### Node 3: `grading` (`rag/nodes/grading.py`)
1. Evaluates all candidate chunks in a **single batch LLM call** using structured output (`BatchGradingResult`).
2. Each chunk receives a score:
   - `relevant`: Contains definitions, code snippets, or context that directly answers the question.
   - `irrelevant`: Tangential keyword hits, different topics, or boilerplate.
3. Filtered chunks form `relevant_documents`.
4. **Fail-Safe Graceful Preservation**: If the grading LLM call encounters a temporary error (e.g. rate limit), it preserves retrieved chunks rather than terminating the pipeline.

#### Node 4: `generate` (`rag/nodes/generation.py`)
1. Formats each verified chunk as a numbered context block: `[Source N: filename — heading]`.
2. Prompts the LLM with strict grounding instructions:
   - Provide a complete, fully articulated answer.
   - Include code blocks (`python`, `bash`, etc.) when present in the context.
   - Insert inline citations `[Source N]` immediately after factual assertions.
   - Refuse to invent details not present in the context.
3. Extracts answer text and compiles structured `SourceMetadata` citations.

#### Node 5: `fallback` (`rag/nodes/fallback.py`)
1. **Greetings**: Returns an orientation message outlining supported documentation topics.
2. **System Errors**: Returns a user-friendly error summary directing the user to check configuration.
3. **General Knowledge Fallback**: When documentation is missing after retry exhaustion, it synthesizes an answer using general LLM knowledge with an explicit disclaimer notice:
   ```markdown
   > ℹ️ **General Technical Knowledge Response**
   > *This topic was not found in your indexed documentation. The following answer is provided from general technical knowledge:*
   ```

---

## 4. Document Ingestion & Chunking Pipeline

### Multi-Format Document Loading (`ingestion/loader.py`)
The system ingests 4 document formats:
- **Markdown (`.md`)**: Preserves document structure and extracts `#` title headers.
- **HTML (`.html`)**: Strips navigation, headers, and scripts using BeautifulSoup, extracting clean content.
- **Plain Text (`.txt`)**: Raw character ingestion.
- **PDF (`.pdf`)**: Extracts page text using pypdf.

### Ingestion Security & SSRF Protection
For URL ingestion (`POST /ingest/url`), the system validates URLs against Server-Side Request Forgery (SSRF):
- Only `http` and `https` schemes are permitted.
- Hostnames resolving to private or local CIDR blocks are blocked:
  - `127.0.0.0/8` (Loopback)
  - `10.0.0.0/8`, `172.16.0.0/12`, `192.168.0.0/16` (Private RFC-1918)
  - `169.254.0.0/16` (Link-local / Cloud metadata)
  - `::1` (IPv6 loopback)

### Hierarchical Markdown-Aware Chunking (`ingestion/chunker.py`)
Technical documentation cannot be split at arbitrary character counts without tearing code blocks away from explanatory text. The chunker implements a hierarchical strategy:
1. **Separators**: Priority order `["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "]`.
2. **Chunk Size**: `800` characters — large enough for a code block and explanation; small enough for tight semantic embedding.
3. **Chunk Overlap**: `100` characters — prevents truncation across boundaries.
4. **Heading Attribution**: Every chunk detects the nearest Markdown header above it and injects it into metadata (`heading`).
5. **SHA-256 Deduplication**: Each chunk computes a SHA-256 hash of its normalized text. When files are re-ingested, identical chunks are skipped automatically, preventing duplicate embeddings.

---

## 5. Question Handling & Supported Query Types

The assistant categorizes user questions into distinct operational types:

### Question Classification Matrix

| Query Type | Typical Intent | Example Question | Processing Path |
|---|---|---|---|
| **`conceptual`** | Understand design philosophy, definitions, architecture | *"What is a StateGraph and how do nodes work?"* | Analysis → Retrieval → Batch Grade → Grounded Generation |
| **`how-to`** | Implementation instructions, syntax, code recipes | *"How do I add a conditional edge in LangGraph?"* | Analysis → Retrieval → Batch Grade → Grounded Generation with code |
| **`troubleshooting`** | Resolving error messages, edge cases, bugs | *"How do I resolve duplicate chunk ID errors in ChromaDB?"* | Analysis → Retrieval → Batch Grade → Grounded Generation |
| **`api-reference`** | Class options, parameters, method signatures | *"What parameters are supported in BaseModel.model_dump?"* | Analysis → Retrieval → Batch Grade → Grounded Generation |
| **`greeting`** | Conversational pleasantry | *"Hello, what can you help me with?"* | Fast-path → Orientation Fallback (Zero retrieval calls) |
| **`out-of-domain`** | Topics completely absent from indexed docs | *"Explain quantum entanglement in physics"* | Analysis → Retrieval → Grading (0 relevant) → Retry 1 → Retry 2 → General Knowledge Fallback |

---

## 6. Real-World Execution Scenarios & Trace Examples

### Scenario 1: How-To Question (Grounded Retrieval)

**User Input:**
```
"How do I add a conditional edge in LangGraph?"
```

**Internal Pipeline Execution:**
1. `query_analysis`: Classified as `how-to`. Rewritten query: `"add conditional edges routing function LangGraph StateGraph"`.
2. `retrieval`: ChromaDB retrieves 5 candidate chunks from `langgraph_concepts.md`.
3. `grading`: Batch grading evaluates all 5 chunks. Chunks 0 and 2 are marked `relevant`; others are marked `irrelevant`.
4. `route_after_grading`: 2 relevant chunks found → routes to `generate`.
5. `generate`: Synthesizes answer with code example and inline citations:
   ```
   In LangGraph, you define conditional routing using `graph_builder.add_conditional_edges()`.
   You specify the source node, a routing function, and a path mapping dictionary [Source 1: langgraph_concepts.md — Edges].
   ```

**API JSON Response (`POST /query`):**
```json
{
  "answer": "In LangGraph, you add a conditional edge using `builder.add_conditional_edges()`. You pass the source node name, a routing function that returns a destination string, and a dictionary mapping return values to node names [Source 1: langgraph_concepts.md — Edges].",
  "sources": [
    {
      "doc_id": "doc_langgraph_concepts",
      "source": "langgraph_concepts.md",
      "chunk_id": "doc_langgraph_concepts_chunk_0003",
      "heading": "Edges"
    }
  ],
  "query_type": "how-to",
  "rewritten_query": "add conditional edges routing function LangGraph StateGraph",
  "retry_count": 0,
  "relevant_chunks_used": 2,
  "generation_supported": true,
  "error": null
}
```

---

### Scenario 2: Terminology Mismatch (Self-Correction Retry Cycle)

**User Input:**
```
"How to persist vectors on hard drive?"
```

**Internal Pipeline Execution:**
1. `query_analysis (retry=0)`: Rewritten to `"persist vectors on hard drive"`.
2. `retrieval`: Returns candidate chunks, but none pass strict relevance grading for the exact phrase.
3. `grading`: 0 chunks deemed relevant.
4. `route_after_grading`: `retry_count (0) < max_retries (2)` → routes back to `query_analysis`.
5. `query_analysis (retry=1)`: Receives feedback that the previous query failed. Generates alternative technical phrasing: `"ChromaDB persist_directory PersistentClient save embeddings to disk"`.
6. `retrieval`: Searches with updated query. Returns 4 chunks from `chromadb_guide.md`.
7. `grading`: 3 chunks verified `relevant`.
8. `route_after_grading`: Relevant docs found → routes to `generate`.
9. `generate`: Produces complete answer explaining ChromaDB persistence with citations.

---

### Scenario 3: Out-of-Domain Query (General Knowledge Fallback)

**User Input:**
```
"Explain quantum entanglement in physics."
```

**Internal Pipeline Execution:**
1. `query_analysis`: Classified as `conceptual`.
2. `retrieval`: ChromaDB returns unrelated technical docs (e.g. FastAPI, LangGraph).
3. `grading`: All chunks graded `irrelevant`.
4. `route_after_grading`: Triggers retry 1, then retry 2.
5. All retries return 0 relevant docs. `retry_count` reaches 2.
6. `route_after_grading`: Retries exhausted → routes to `fallback`.
7. `fallback`: Detects documentation absence. Formulates general technical response with warning disclaimer:

```markdown
> ℹ️ **General Technical Knowledge Response**
> *This topic was not found in your indexed documentation. The following answer is provided from general technical knowledge:*

Quantum entanglement is a physical phenomenon that occurs when a group of particles interact in ways such that the quantum state of each particle cannot be described independently of the state of the others...
```

---

### Scenario 4: Fast-Path Conversational Greeting

**User Input:**
```
"Hello!"
```

**Internal Pipeline Execution:**
1. `query_analysis`: Fast-path token inspection detects greeting. Sets `query_type = "greeting"`.
2. `route_after_query_analysis`: Routes directly to `fallback`.
3. `fallback`: Immediately responds with assistant overview and suggested topics.
4. **Latency**: Sub-50ms (zero database lookups, zero LLM grading calls).

---

## 7. Architectural Q&A & Technical Interview Deep-Dive

### Q1: Why LangGraph instead of standard LangChain chains or sequential pipelines?
**Answer**:
Sequential chains (like basic LCEL or `RetrievalQA`) are strictly acyclic: if the retrieval step returns poor context, the system is doomed to pass bad context to the generator. LangGraph models the workflow as a **cyclic state graph (`StateGraph`)**. This enables:
- Conditional branching based on intermediate runtime state (e.g. routing greetings away from vector search).
- Self-correction loops where failed retrievals cycle back to reformulate queries.
- Clean separation of concerns into modular, independently testable nodes (`query_analysis`, `retrieval`, `grading`, `generate`, `fallback`).

---

### Q2: Why cap `max_retries` at 2?
**Answer**:
Every retry incurs an LLM query analysis call and a vector search. On a specialized local documentation corpus:
- Retry 1 handles vocabulary mismatch (e.g., colloquial terms vs. formal documentation terms).
- Retry 2 tries a broader conceptual query.
- A 3rd attempt on a defined corpus has diminishing returns (<5% chance of finding unindexed data) while adding 3–5 seconds of user-facing latency. A cap of 2 balances recall against responsiveness.

---

### Q3: How is hallucination prevented in generated answers?
**Answer**:
Hallucination prevention is enforced through a multi-layered defense:
1. **Relevance Grading Gate**: Every chunk retrieved must pass the `grading` node before reaching the generator. Irrelevant or noisy chunks are discarded.
2. **Strict System Prompt Constraints**: The generator prompt explicitly commands the model to use *only* facts present in the numbered context blocks and to include inline citations.
3. **Structured Citations**: The generation node constructs verified `SourceMetadata` objects containing the `doc_id`, `chunk_id`, filename, and section heading.
4. **Explicit Fallback**: When retries are exhausted, the model does not attempt to fabricate answers from non-existent documentation; it explicitly informs the user that the information is from general knowledge.

---

### Q4: Why use `sentence-transformers/all-MiniLM-L6-v2` locally instead of OpenAI or Gemini embeddings?
**Answer**:
1. **Zero Operational Cost**: Local embeddings run on CPU without consuming API credits or rate limits.
2. **Zero External Network Latency**: Vector lookups during indexing and query retrieval happen entirely in-memory and on disk.
3. **Privacy & Offline Operation**: Document contents never leave the server during vector indexing.
4. **Proven Performance**: `all-MiniLM-L6-v2` generates 384-dimensional dense vectors with exceptional performance on technical prose and semantic retrieval.

---

### Q5: Why ChromaDB over FAISS or cloud vector databases (Pinecone, Weaviate)?
**Answer**:
1. **Built-in Metadata Storage & Filtering**: ChromaDB stores document text, chunk IDs, source filenames, and section headings directly in the collection, whereas FAISS requires maintaining a separate metadata lookup table.
2. **Persistence Without Complexity**: Simple on-disk storage (`./chroma_db`) that survives application restarts without requiring external server processes.
3. **Document Management Operations**: ChromaDB natively supports listing documents, filtering by `doc_id`, and targeted deletion needed for `GET /documents` and `DELETE /documents/{doc_id}`.

---

### Q6: How does batch document grading work, and why not grade chunks individually?
**Answer**:
Individual chunk grading requires $N$ sequential LLM requests for $N$ retrieved chunks. If $N = 5$, this adds 5 network round-trips (~3–4 seconds).
Batch grading concatenates all candidate chunks into a single structured prompt and asks the LLM for a `BatchGradingResult` containing an array of evaluations. This achieves:
- **80% Reduction in Grading Latency**: 1 network round-trip instead of 5.
- **Cross-Chunk Comparative Context**: The LLM can assess which chunks provide complementary vs. redundant information.
- **Lower Token Overhead**: The system instructions are transmitted once rather than repeatedly.

---

### Q7: How does chunk deduplication work during continuous ingestion?
**Answer**:
During chunking, the system computes a SHA-256 hash of each chunk's normalized text and stores it in the chunk's metadata as `content_hash`.
When documents are added via `add_documents()`:
1. ChromaDB is queried for existing `content_hash` values in the collection.
2. Incoming chunks whose hashes already exist in the database are filtered out.
3. Only genuinely new or modified chunks are embedded and inserted.
This prevents vector bloat and avoids duplicate search results when files are uploaded multiple times.

---

### Q8: How is user feedback handled and what is its operational purpose?
**Answer**:
User ratings (`thumbs_up` / `thumbs_down`) and comments submitted via the UI or `POST /feedback` are appended to an atomic, durable JSON Lines file (`data/feedback.jsonl`).
Each entry contains:
- Unique feedback ID (`fb_...`)
- Timestamp (UTC ISO-8601)
- User rating and comment
- The original user question
- The API's answered response and cited sources
This log serves as an **offline evaluation dataset** for fine-tuning prompt templates, diagnosing retrieval failures, and identifying gaps in the documentation corpus.

---

## 8. Complete API Endpoint Reference

All endpoints run on `http://localhost:8000` with interactive Swagger UI at `/docs`.

### 1. `POST /query`
Executes the self-corrective LangGraph pipeline.

- **Request Body**:
  ```json
  {
    "question": "string (min 3 chars, max 1000 chars)",
    "top_k": 5
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "answer": "string",
    "sources": [
      {
        "doc_id": "string",
        "source": "string",
        "chunk_id": "string",
        "heading": "string"
      }
    ],
    "query_type": "conceptual | how-to | troubleshooting | api-reference | greeting",
    "rewritten_query": "string",
    "retry_count": 0,
    "relevant_chunks_used": 2,
    "generation_supported": true,
    "error": null
  }
  ```

---

### 2. `POST /ingest`
Uploads and indexes local documentation files (`.md`, `.txt`, `.html`, `.pdf`).

- **Content-Type**: `multipart/form-data`
- **Form Data**: `files` (array of uploaded binary files)
- **Response (200 OK)**:
  ```json
  {
    "ingested": [
      {
        "source": "guide.md",
        "doc_id": "doc_guide",
        "chunks_added": 12,
        "chunks_skipped_duplicate": 0
      }
    ],
    "errors": [],
    "total_chunks_added": 12
  }
  ```

---

### 3. `POST /ingest/url`
Fetches and indexes documentation from external web URLs.

- **Request Body**:
  ```json
  {
    "urls": ["https://raw.githubusercontent.com/.../README.md"]
  }
  ```
- **Response (200 OK)**: Same schema as `POST /ingest`.

---

### 4. `GET /documents`
Lists all documents currently indexed in ChromaDB.

- **Response (200 OK)**:
  ```json
  {
    "total_documents": 5,
    "total_chunks": 94,
    "documents": [
      {
        "doc_id": "doc_fastapi_guide",
        "source": "fastapi_guide.md",
        "title": "FastAPI Core Guide",
        "chunks": 21,
        "ingested_at": "2026-11-01T10:00:00Z"
      }
    ]
  }
  ```

---

### 5. `DELETE /documents/{doc_id}`
Deletes a specific document and all of its associated chunks from ChromaDB.

- **Response (200 OK)**:
  ```json
  {
    "status": "deleted",
    "doc_id": "doc_fastapi_guide",
    "chunks_removed": 21
  }
  ```

---

### 6. `DELETE /documents`
Wipes all documents and resets the vector store.

- **Response (200 OK)**:
  ```json
  {
    "status": "cleared",
    "message": "All documents removed from vector store."
  }
  ```

---

### 7. `POST /feedback`
Records thumbs-up/down ratings and user comments.

- **Request Body**:
  ```json
  {
    "rating": "thumbs_up | thumbs_down",
    "comment": "string (optional)",
    "question": "string"
  }
  ```
- **Response (200 OK)**:
  ```json
  {
    "status": "recorded",
    "feedback_id": "fb_a1b2c3d4e5f6",
    "message": "Thank you for your feedback."
  }
  ```

---

### 8. `GET /health`
Returns system operational status, model configuration, and indexed counts.

- **Response (200 OK)**:
  ```json
  {
    "status": "healthy",
    "llm_configured": true,
    "llm_model": "gemini-3.8-flash",
    "indexed_chunks": 94,
    "indexed_documents": 5
  }
  ```

---

## 9. Developer Setup & Verification Guide

### 1. Environment Configuration

Create a `.env` file from `.env.example`:
```bash
cp .env.example .env
```

Set your configuration values:
```env
GOOGLE_API_KEY=your_actual_gemini_api_key
LLM_MODEL=gemini-3.8-flash
LLM_TEMPERATURE=0.0
CHROMA_PERSIST_DIR=./chroma_db
CHROMA_COLLECTION_NAME=tech_docs
EMBEDDING_MODEL=all-MiniLM-L6-v2
RETRIEVAL_TOP_K=5
MAX_RETRIES=2
CHUNK_SIZE=800
CHUNK_OVERLAP=100
AUTO_INGEST_CORPUS=true
```

### 2. Run the FastAPI Backend

```bash
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive API documentation available at: `http://127.0.0.1:8000/docs`
- Health check available at: `http://127.0.0.1:8000/health`

### 3. Run the Streamlit Frontend

In a separate terminal window:
```bash
streamlit run ui.py
```
- Web interface will open at: `http://localhost:8501`

### 4. Run the Test Suite

```bash
pytest tests/ -v
```
All unit tests covering loaders, chunking, SSRF filters, graph nodes, state routing, and FastAPI endpoints will execute.
