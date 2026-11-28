"""
ModelShield Adaptive Red-Team Orchestrator - REST API
Provides endpoints to initiate runs, inspect progress, retrieve evidence, and query the copilot.
"""
import json
import os
from typing import Any, Dict, List, Optional

try:
    from fastapi import APIRouter, HTTPException, Query
    from pydantic import BaseModel, Field
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


if FASTAPI_AVAILABLE:
    router = APIRouter(prefix="/api/red-team", tags=["Adaptive Red-Team Orchestrator"])

    class RunRequest(BaseModel):
        sample_size: int = Field(default=32, ge=4, le=128)
        max_budget: int = Field(default=5, ge=2, le=10)
        epsilon: float = Field(default=8 / 255, gt=0.0, le=0.5)
        pgd_steps: int = Field(default=10, ge=1, le=50)
        pgd_alpha: float = Field(default=2 / 255, gt=0.0, le=0.5)
        quantization_levels: int = Field(default=16, ge=2, le=256)

    class CopilotInquiryRequest(BaseModel):
        run_id: str
        query: str

    @router.get("/health")
    def health_check():
        return {"status": "ok", "service": "ModelShield Red-Team Orchestrator"}

    @router.get("/runs")
    def list_runs(results_dir: str = "results/runs") -> List[Dict[str, Any]]:
        if not os.path.exists(results_dir):
            return []
        runs = []
        for fname in sorted(os.listdir(results_dir), reverse=True):
            if fname.endswith(".json"):
                fpath = os.path.join(results_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    runs.append({
                        "run_id": data.get("run_id"),
                        "model_name": data.get("model_name"),
                        "status": data.get("final_assessment", {}).get("overall_status"),
                        "clean_accuracy": data.get("final_assessment", {}).get("clean_accuracy"),
                        "max_asr": data.get("final_assessment", {}).get("max_attack_success_rate"),
                        "budget_consumed": data.get("final_assessment", {}).get("budget_consumed"),
                        "timestamp": data.get("final_assessment", {}).get("timestamp"),
                    })
                except Exception:
                    continue
        return runs

    @router.get("/runs/{run_id}")
    def get_run(run_id: str, results_dir: str = "results/runs") -> Dict[str, Any]:
        fpath = os.path.join(results_dir, f"{run_id}.json")
        if not os.path.exists(fpath):
            raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)

    @router.get("/runs/{run_id}/assessment")
    def get_run_assessment(run_id: str, results_dir: str = "results/runs") -> Dict[str, Any]:
        run_data = get_run(run_id, results_dir)
        assessment = run_data.get("final_assessment")
        if not assessment:
            raise HTTPException(status_code=404, detail="Assessment not available for this run.")
        return assessment

    @router.get("/runs/{run_id}/decisions")
    def get_run_decisions(run_id: str, results_dir: str = "results/runs") -> List[Dict[str, Any]]:
        run_data = get_run(run_id, results_dir)
        return run_data.get("decisions", [])

    @router.post("/copilot/inquire")
    def copilot_inquire(req: CopilotInquiryRequest, results_dir: str = "results/runs") -> Dict[str, str]:
        from orchestrator.copilot import InvestigationCopilot
        run_data = get_run(req.run_id, results_dir)
        copilot = InvestigationCopilot(run_data)
        answer = copilot.answer_inquiry(req.query)
        return {"run_id": req.run_id, "query": req.query, "answer": answer}

