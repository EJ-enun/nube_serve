import os
import requests
from fastapi import FastAPI
from fastmcp import FastMCP, Context
from typing import Dict, Any
from fastapi import Request, HTTPException
app = FastAPI(title="Fourth Trimester Care Agent")
mcp = FastMCP("FourthTrimester")

# --- HUGGING FACE CONFIGURATION ---
HF_API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.environ.get("HF_TOKEN") # Set this in Render Env Vars
MODEL_ID = "google/gemma-2-2b-it:featherless-ai"

class MaternalHealthIntelligence:
    """Logic for identifying the 12-week care gap and risk markers."""

    @staticmethod
    def query_gemma(prompt: str) -> Dict[str, Any]:
        """Calls the Hugging Face Inference API."""
        headers = {"Authorization": f"Bearer {HF_TOKEN}"}
        payload = {
            "model": MODEL_ID,
            "messages": [
                {"role": "system", "content": "You are a Maternal Health Equity Agent. Focus on the 12-week postpartum period for high-risk Black women."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1 # Keep it clinical and consistent
        }
        
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        return response.json()

# --- MCP TOOLS ---

@app.middleware("http")
async def validate_api_key(request: Request, call_next):
    # This checks the header you configured in the Prompt Opinion dashboard
    api_key = request.headers.get("X-API-Key")
    if api_key != "maternity-secret-2026": # Use the value you typed in the modal
        raise HTTPException(status_code=403, detail="Unauthorized MCP Access")
    return await call_next(request)
    
@mcp.tool()
async def audit_postpartum_gap(ctx: Context, fhir_json: str) -> str:
    """
    Analyzes FHIR data to identify Black patients within 12 weeks of delivery 
    with high-risk markers and no scheduled follow-up.
    """
    
    # 1. Construct the prompt for the model using the optimized Lyra logic
    prompt = f"""
    AUDIT TASK: Analyze the following patient data for a 'Fourth Trimester' care gap.
    
    DATA:
    {fhir_json}
    
    REQUIREMENTS:
    1. Identify high-risk markers (Preeclampsia, Hypertension, etc.).
    2. Confirm if a postpartum follow-up is missing.
    3. Provide clinical reasoning and a DIRECT outreach message for the coordinator.
    
    Return your response as a clear clinical report.
    """

    # 2. Call the Hugging Face Model
    intelligence = MaternalHealthIntelligence()
    hf_response = intelligence.query_gemma(prompt)
    
    # 3. Extract the model's text
    try:
        report = hf_response["choices"][0]["message"]["content"]
    except KeyError:
        report = "Error: Could not retrieve clinical audit from model backend."

    return f"--- FOURTH TRIMESTER CLINICAL AUDIT ---\n{report}"

# --- A2A DISCOVERY (For Render Deployment) ---

@app.get("/.well-known/agent.json")
async def get_agent_card():
    host = os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'localhost:8000')
    protocol = "https" if "render" in host else "http"
    
    return {
        "name": "Fourth Trimester Agent",
        "description": "Closes the 12-week postpartum care gap for high-risk patients.",
        "endpoint": f"{protocol}://{host}/mcp",
        "capabilities": {"inter_agent_chat": True},
        "skills": ["Maternal Health Equity", "Postpartum Gap Analysis", "Clinical Outreach"]
    }
    
app.mount("/mcp", mcp)