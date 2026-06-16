import asyncio

from loguru import logger

from app.database import init_db, add_ticket_embedding


TICKETS = [
    {
        "ticket_id": "A0809-679",
        "title": "[RAG] Add OCR for automatic document chunking",
        "description": "Maybe we can add OCR system for better document chunking. References: tesseract, deepseek-ocr",
    },
    {
        "ticket_id": "A0809-460",
        "title": "[RAG] Implement Reranking System for RAG Pipeline",
        "description": "Reranking described at pinecone.io/learn/series/rag/rerankers. Implementing that in our environment should improve retrieved information",
    },
    {
        "ticket_id": "A0809-512",
        "title": "[RAG] Evaluate embedding models for retrieval",
        "description": "Current embedding model may not be optimal for technical documentation. We should benchmark nomic-embed-text, bge-m3 and other candidates on our internal dataset and pick the best one based on retrieval metrics.",
    },
    {
        "ticket_id": "A0809-301",
        "title": "[INFRA] Migrate CI/CD pipeline to GitLab",
        "description": "Our current Jenkins setup is outdated and hard to maintain. We should migrate all pipelines to GitLab CI/CD to align with company standards and improve developer experience.",
    },
]


async def main():
    await init_db()
    for ticket in TICKETS:
        text = f"{ticket['title']} {ticket['description']}"
        await add_ticket_embedding(ticket["ticket_id"], text)
        logger.info(f"✅ Added embedding for {ticket['ticket_id']}")
    logger.info("Done!")


if __name__ == "__main__":
    asyncio.run(main())
