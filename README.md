# ModelShield

An adversarial robustness evaluation platform for image-classification models, featuring attack testing, evaluation reliability checks, and security regression analysis.

## Project Structure

```text
ModelShield/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # Application entrypoint (e.g. FastAPI)
│   │   ├── api/               # API routes and endpoints
│   │   ├── core/              # Config, security, and application settings
│   │   ├── schemas/           # Pydantic schemas / request-response models
│   │   ├── services/          # Core business and model inspection logic
│   │   └── db/                # Database models and session management
│   ├── tests/                 # Unit and integration test suites
│   └── requirements.txt       # Python dependencies
├── docs/
│   └── architecture.md        # Architectural overview and design documentation
├── .gitignore
└── README.md
```

## Getting Started

### 1. Set up a Virtual Environment
```bash
cd backend
python -m venv venv
# On Windows PowerShell:
.\venv\Scripts\Activate.ps1
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Backend Application
```bash
uvicorn app.main:app --reload
```
