from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request

from app.database import close_db, init_db
from app.graph import create_graph
from app.state import InputState


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    app.state.graph = create_graph()
    yield
    await close_db()


app = FastAPI(lifespan=lifespan)


@app.get("/")
async def home():
    return {"status": "ok"}


@app.post("/tickets")
async def add_ticket(input_state: InputState, request: Request):
    graph = request.app.state.graph

    config = {"configurable": {"thread_id": input_state.ticket_id}}

    await graph.ainvoke(
        {"input_state": input_state},
        config=config,
    )
    return {
        "status": "processing started",
        "ticket_id": input_state.ticket_id,
    }


@app.get("/tickets/{ticket_id}")
async def get_ticket(ticket_id: str, request: Request):
    graph = request.app.state.graph

    config = {"configurable": {"thread_id": ticket_id}}
    state = await graph.aget_state(config)

    if not state.values:
        raise HTTPException(status_code=404, detail="Ticket not found")

    return {
        "status": "ok",
        "ticket_id": ticket_id,
        "state": state.values,
    }


@app.post("/tickets/{ticket_id}/retry")
async def retry_ticket(ticket_id: str, request: Request):
    graph = request.app.state.graph

    config = {"configurable": {"thread_id": ticket_id}}
    state = await graph.aget_state(config)

    if not state.values:
        raise HTTPException(status_code=404, detail="Ticket not found")

    await graph.ainvoke(None, config=config)

    return {
        "status": "retried",
        "ticket_id": ticket_id,
    }
