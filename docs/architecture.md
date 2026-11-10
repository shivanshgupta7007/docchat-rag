# Architecture

```
PDF upload
   |
   v
extract_pdf_pages (pypdf)  --> [(page, text)]
   |
   v
chunk_document (paragraph packing, overlap, page metadata)
   |
   v
GeminiEmbedder.embed (retrieval_document) --> vectors
   |
   v
VectorStore (ChromaDB, cosine)  <--------------------+
                                                     |
question --> GeminiEmbedder.embed_query --> query ---+--> top-k chunks
                                                     |
                                                     v
                                        build_prompt (numbered, cited)
                                                     |
                                                     v
                                        GeminiChat.generate --> answer + sources
```
