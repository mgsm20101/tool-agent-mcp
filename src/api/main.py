"""FastAPI front for the agent.

POST /run  { "task": "..." }  ->  answer + the full trace (the trace is the product:
it shows think/act/observe and any guardrail interventions step by step).
"""

from __future__ import annotations

from dataclasses import asdict

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.config import settings

app = FastAPI(title="Tool Agent (MCP)", version="0.1.0")


class TaskRequest(BaseModel):
    task: str = Field(..., min_length=2)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": settings.llm_model, "max_iterations": settings.max_iterations}


@app.post("/run")
async def run(req: TaskRequest) -> dict:
    try:
        from src.agent.loop import run_agent

        result = await run_agent(req.task)
    except Exception as e:  # noqa: BLE001 - surface MCP/LLM failures as 502
        raise HTTPException(status_code=502, detail=f"agent error: {e}") from e
    return {
        "task": req.task,
        "answer": result.answer,
        "iterations": result.iterations,
        "tools_used": result.tools_used,
        "stopped_by": result.stopped_by,
        "trace": [asdict(s) for s in result.trace],
    }
