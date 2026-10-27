from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="ModelShield API",
    description="Security, Guardrails, and Monitoring for AI/ML Inference Pipelines",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {
        "service": "ModelShield API",
        "status": "healthy",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {"status": "ok"}
