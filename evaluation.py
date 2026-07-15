"""
G-Eval based scoring for assistant responses.

Uses deepeval's GEval metric (LLM-as-judge, chain-of-thought / form-filling
scoring paradigm) to score responses on Answer Relevancy, Faithfulness, and
Coherence. Designed to be called from a FastAPI background task after a live
response is stored, or from an offline backfill script - never on the
request/response latency path.
"""

import os
import logging
import uuid
from typing import Any, Dict, List, Optional

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, SingleTurnParams

from database import SessionLocal
from schema2_crud import insert_response_evaluation

logger = logging.getLogger(__name__)

EVAL_MODEL = os.getenv("OPENAI_EVAL_MODEL", os.getenv("OPENAI_MODEL", "gpt-4o-mini"))


def build_geval_metrics() -> Dict[str, GEval]:
    """Construct the three G-Eval metrics used to score assistant responses."""
    relevancy = GEval(
        name="Answer Relevancy",
        criteria=(
            "Determine whether the actual output directly and completely "
            "addresses the question asked in the input, without going off-topic."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=EVAL_MODEL,
    )

    faithfulness = GEval(
        name="Faithfulness",
        criteria=(
            "Determine whether every factual claim in the actual output is "
            "supported by the retrieval context, with no hallucinated or "
            "unsupported information."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.RETRIEVAL_CONTEXT,
        ],
        model=EVAL_MODEL,
    )

    coherence = GEval(
        name="Coherence",
        criteria=(
            "Determine whether the actual output is logically organized, "
            "internally consistent, and clearly written."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=EVAL_MODEL,
    )

    return {
        "relevancy": relevancy,
        "faithfulness": faithfulness,
        "coherence": coherence,
    }


_METRICS = build_geval_metrics()


def evaluate_response(
    query: str,
    answer: str,
    retrieval_context: Optional[List[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Score an assistant response with G-Eval.

    Faithfulness is skipped when retrieval_context is empty, since there is
    nothing to ground the response against.

    Returns a dict keyed by metric name, each value {"score": float|None, "reason": str|None}.
    A metric that raises is recorded as {"score": None, "reason": None} and logged,
    never propagated - a scoring failure must never affect the caller.
    """
    retrieval_context = retrieval_context or []
    test_case = LLMTestCase(
        input=query,
        actual_output=answer,
        retrieval_context=retrieval_context if retrieval_context else None,
    )

    results: Dict[str, Dict[str, Any]] = {}
    for metric_name, metric in _METRICS.items():
        if metric_name == "faithfulness" and not retrieval_context:
            continue
        try:
            metric.measure(test_case)
            results[metric_name] = {"score": metric.score, "reason": metric.reason}
        except Exception as e:
            logger.warning(f"[EVAL] {metric_name} scoring failed: {e}")
            results[metric_name] = {"score": None, "reason": None}

    return results


def run_and_store_evaluation(
    message_id: uuid.UUID,
    conversation_id: uuid.UUID,
    query: str,
    answer: str,
    retrieval_context: Optional[List[str]] = None,
) -> None:
    """
    Entry point for the live background-task hook. Opens its own DB session
    (the request-scoped session may already be closed by the time this runs),
    scores the response, and persists the result. Never raises.
    """
    db = SessionLocal()
    try:
        results = evaluate_response(query, answer, retrieval_context)
        insert_response_evaluation(db, message_id, conversation_id, results)
    except Exception as e:
        logger.warning(f"[EVAL] Failed to store evaluation for message {message_id}: {e}")
    finally:
        db.close()
