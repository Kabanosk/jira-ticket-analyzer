from psycopg_pool import AsyncConnectionPool
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from pydantic import computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict
from langchain_ollama import OllamaEmbeddings

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")

    postgres_user: str
    postgres_password: str
    postgres_db: str
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    embeddings_base_url: str = "http://localhost:11434"
    embeddings_api_key: str = "ollama"
    embeddings_model: str = "nomic-embed-text"

    llm_model: str = "gemma3:1b"
    llm_base_url: str = "http://localhost:11434"

    @computed_field
    @property
    def database_url(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


settings = Settings()
embeddings = OllamaEmbeddings(
    model=settings.embeddings_model,
    base_url=settings.embeddings_base_url
)
pool: AsyncConnectionPool
checkpointer: AsyncPostgresSaver


async def create_tables():
    async with pool.connection() as conn:
        await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
        await conn.execute(
            "CREATE TABLE IF NOT EXISTS tickets (ticket_id TEXT PRIMARY KEY,text TEXT,embedding vector(768))"
        )



async def init_db() -> None:
    global pool, checkpointer
    pool = AsyncConnectionPool(conninfo=settings.database_url)
    await pool.open()
    checkpointer = AsyncPostgresSaver(pool)
    await checkpointer.setup()
    await create_tables()

async def close_db() -> None:
    assert pool is not None, "Pool not initialized"
    await pool.close()


async def search_similar_tickets(text: str) -> list[str]:
    embedding: list[float] = await embeddings.aembed_query(text)
    async with pool.connection() as conn:
        cursor = await conn.execute(
            "SELECT ticket_id FROM tickets ORDER BY embedding <=> %s::vector LIMIT 3", (embedding,)
        )
        results = await cursor.fetchall()
    return [ticket_id for (ticket_id,) in results]
