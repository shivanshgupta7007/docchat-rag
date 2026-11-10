# DocChat - How to Explain It (Study Guide)

Read this until you can say it in your own words. Interviewers will probe until they find the edge of your understanding - these notes mark where the edges are.

## The 30-second version

"DocChat lets you upload a PDF and chat with it. The problem with asking an LLM about a document directly is context limits and hallucination. So instead, the app indexes the document once: it splits it into small overlapping chunks, turns each chunk into an embedding vector, and stores them in ChromaDB. When you ask a question, it embeds the question, pulls the 4 most similar chunks, and has Gemini answer using only those chunks, with citations. So every answer traces back to an actual page."

## End-to-end flow (be able to draw this)

1. **Extract**: pypdf pulls text page by page, keeping page numbers for citations.
2. **Chunk**: paragraph-packing splitter, ~800 characters per chunk with 150 characters of overlap. Why overlap? So a sentence cut in half at a boundary still appears complete in the next chunk.
3. **Embed**: Gemini `text-embedding-004` converts each chunk into a vector - a point in meaning-space, so similar meanings sit close together.
4. **Store**: ChromaDB keeps vectors + metadata (source file, page). Cosine similarity.
5. **Retrieve**: the question gets its own embedding (`retrieval_query` task type) and we take the top-4 nearest chunks.
6. **Generate**: a prompt template numbers the chunks [1]..[4], instructs Gemini to answer only from them, and to say "I couldn't find that" when the context is insufficient. Temperature 0.2 - we want fidelity, not creativity.
7. **Respond**: the API returns the answer plus the sources (file, page, similarity score, snippet) so the UI can show exactly where each answer came from.

## Key code to know cold

- `app/chunking.py` -> `chunk_text()`: the greedy packer with overlap. Most-asked-about file.
- `app/rag.py` -> `build_prompt()` and `RAGPipeline.ask()`: the actual R in RAG.
- `app/store.py` -> `VectorStore.query()`: how similarity search works.

## Why these choices (the "why" questions)

- **Why chunk at all?** LLM context is finite and attention degrades over long inputs; small relevant chunks give better answers and cost fewer tokens.
- **Why 800 characters?** Roughly 200 tokens - small enough for precise retrieval, big enough to keep a full idea. It's a tunable trade-off.
- **Why a vector DB instead of keyword search?** "How do I get my money back?" matches "refund policy" with zero shared words. Embeddings capture meaning; keywords capture spelling.
- **Why cosine similarity?** It measures direction, not magnitude, so a long chunk and a short chunk about the same thing still match.
- **Why citations?** Trust. A RAG system's main job is to make answers verifiable, which is what separates it from plain chatbot hallucination.
- **Why FastAPI + Streamlit?** FastAPI is the production interface other systems would call; Streamlit is a fast demo UI. Separation of concerns.

## Likely interview questions

**Q: What happens if the answer isn't in the document?**
The prompt explicitly instructs the model to say it couldn't find the answer rather than invent one. Grounding plus that instruction is the main hallucination defense.

**Q: How would you improve retrieval quality?**
Start with chunk-size tuning and larger top-k with a re-ranker on top (cross-encoder). Then hybrid search: combine BM25 keyword scores with vector scores. Finally, query rewriting for vague questions.

**Q: How would you evaluate this system?**
Build a set of question/expected-answer pairs from real documents. Measure retrieval hit rate (did the right chunk come back in top-k?) and answer faithfulness (does the answer only use retrieved content?). An LLM-as-judge with a rubric works for faithfulness at scale - which is exactly what my second project, the agent eval harness, implements.

**Q: What breaks at scale?**
Chroma is embedded and single-node; at millions of chunks you'd move to Pinecone/Weaviate/pgvector. Ingestion becomes a background job queue, and you'd add caching of embeddings and answers.

**Q: Why not fine-tune a model on the documents instead?**
Fine-tuning bakes in static knowledge, is expensive to refresh, and can't cite sources. RAG updates by re-indexing a file and keeps every answer traceable. Fine-tuning is for style/behavior, RAG is for knowledge.

**Q: What's asymmetric retrieval?**
Embedding queries and documents with different task-type hints (`retrieval_query` vs `retrieval_document`). The model is trained to use that asymmetry, and recall improves measurably.
