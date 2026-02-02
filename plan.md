# LightSearch

LightSearch is a tiny, SQLite-backed search library for small, local
projects.

It provides: - Persistent text search using SQLite FTS5 - Persistent
vector search using LSH (random projections) + exact reranking -
Optional hybrid search by combining text and vector results - A
single-file `.db` storage model

LightSearch is designed for pet projects, demos, prototypes, and course
projects where you want real search functionality without running any
external services.

## Relationship to minsearch

-   minsearch
    -   in-memory only
    -   extremely simple
    -   no persistence
    -   no LSH / ANN
    -   great for experiments and notebooks
-   LightSearch
    -   persistent (SQLite file)
    -   still simple, still local
    -   supports text search (FTS5)
    -   supports vector search (LSH + rerank)
    -   supports hybrid search

Rule of thumb: - If everything fits in memory and you don't need
persistence → minsearch - If you want persistence but still zero
infrastructure → LightSearch

LightSearch is a persistent, SQLite-based sibling of minsearch, not a
replacement for real search engines.

## When should you use LightSearch?

Use LightSearch when: - you want zero infrastructure - you don't want to
run Postgres, Elasticsearch, OpenSearch, or a vector DB - you're
building a pet project, hackathon demo, course assignment, or early
prototype - your dataset is small (up to \~10--20k documents) - query
volume is low to moderate - you value simplicity over maximum
performance

## When should you NOT use LightSearch?

LightSearch is not recommended for production workloads.

Avoid it when you need: - high-QPS or low-latency guarantees -
concurrent multi-user workloads - advanced ranking or learning-to-rank -
large-scale vector search (100k+ vectors) - distributed storage or
replication - operational reliability and SLOs

For production systems, use: - Text search: Elasticsearch, OpenSearch,
Meilisearch, Typesense - Vector search: Qdrant, Milvus, Weaviate,
pgvector/Postgres

## Architecture overview

LightSearch provides two independent index classes: - TextSearchIndex -
VectorSearchIndex

They can operate in two modes.

### Mode 1: Shared database (hybrid search)

Both indexes use the same SQLite database file and share a common `docs`
table.

This enables hybrid search: - text search via FTS5 - vector search via
LSH - results combined at the application level

This is the recommended mode if you want text and vector search over the
same documents.

### Mode 2: Independent databases (full isolation)

Each index uses its own SQLite file.

Use this when: - you want complete isolation - you don't care about
hybrid search - you want maximum conceptual simplicity

## Design goals

LightSearch intentionally: - avoids SQLite extensions - avoids native
code - avoids background services - avoids complex ANN algorithms

Instead, it focuses on: - correctness - transparency - debuggability -
ease of understanding

LSH is used only to narrow candidates. Final ranking is always done with
exact cosine similarity in NumPy.

## Summary

LightSearch sits in a very specific niche:

"I want real search, persistence, and vectors --- but I don't want to
run any infrastructure yet."

It complements minsearch and provides a natural upgrade path: minsearch
→ LightSearch → production-grade search engines
