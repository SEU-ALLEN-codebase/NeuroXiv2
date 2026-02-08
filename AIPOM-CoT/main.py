import logging
import os
import sys
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field
import uvicorn

from aipom_v10_production import AIPOMCoTV10  # noqa: E402


BASE_DIR = Path(__file__).resolve().parent

logger = logging.getLogger("aipom_api")


class AnswerRequest(BaseModel):
    question: str = Field(..., description="Natural language question for the agent")
    max_iterations: int = Field(15, ge=1, le=30, description="Safety cap for agent loop")
    generate_plots: bool = Field(False, description="If true, also run Figure4 visualization pipeline")
    output_dir: Optional[str] = Field(None, description="Optional override for plot output directory")


class AnswerResponse(BaseModel):
    success: bool
    data: dict


app = FastAPI(title="AIPOM-CoT API", version="1.0.0")

_agent: Optional[AIPOMCoTV10] = None


def _build_agent() -> AIPOMCoTV10:
    """Create a shared agent instance using environment-driven config."""
    schema_path = BASE_DIR / "schema_output" / "schema.json"

    return AIPOMCoTV10(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_pwd=os.getenv("NEO4J_PASSWORD", "neuroxiv"),
        database=os.getenv("NEO4J_DATABASE", "neo4j"),
        schema_json_path=str(schema_path),
        openai_api_key=os.getenv("OPENAI_API_KEY",''),
        model=os.getenv("OPENAI_MODEL", "gpt-4o"),
    )


@app.on_event("startup")
def _startup_event() -> None:
    global _agent

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        _agent = _build_agent()
        logger.info("AIPOM-CoT agent initialized for API usage")
    except Exception:  # noqa: BLE001
        logger.exception("Failed to initialize AIPOM-CoT agent on startup")
        raise


@app.on_event("shutdown")
def _shutdown_event() -> None:
    if _agent is not None:
        try:
            _agent.close()
        except Exception:  # noqa: BLE001
            logger.exception("Error while closing AIPOM-CoT agent")


@app.get("/health")
async def health() -> dict:
    """Health probe for orchestrators."""
    return {"status": "ok", "agent_ready": _agent is not None}


@app.post("/answer", response_model=AnswerResponse)
async def answer(payload: AnswerRequest) -> AnswerResponse:
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent not initialized")

    try:
        result = (
            _agent.answer_with_visualization(
                payload.question,
                max_iterations=payload.max_iterations,
                generate_plots=payload.generate_plots,
                output_dir=payload.output_dir or str(BASE_DIR / "figure4_api_output"),
            )
            if payload.generate_plots
            else _agent.answer(payload.question, max_iterations=payload.max_iterations)
        )

        return AnswerResponse(success=True, data=jsonable_encoder(result))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent failed to answer the question")
        raise HTTPException(status_code=500, detail=str(exc))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "5454")),
        reload=False,
    )
