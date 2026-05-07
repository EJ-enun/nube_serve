import os
import requests
from fastapi import FastAPI, Request, HTTPException
from fastmcp import FastMCP, Context
from typing import Dict, Any

# 1. Initialize MCP and FastAPI
mcp = FastMCP("FourthTrimester")
app = FastAPI(title="Fourth Trimester Care Agent")


# --- CLINICAL LOGIC ---
class MaternalHealthIntelligence:
    @staticmethod
    def query_gemma(prompt: str) -> str:
        token = os.environ.get("HF_TOKEN")
        headers = {"Authorization": f"Bearer {token}"}
        # Using the standard Inference API endpoint
        api_url = "https://api-inference.huggingface.co/models/google/gemma-2-2b-it"
        
        payload = {
            "inputs": prompt,
            "parameters": {"max_new_tokens": 500, "temperature": 0.1}
        }
        
        try:
            response = requests.post(api_url, headers=headers, json=payload, timeout=20)
            if response.status_code == 200:
                res = response.json()
                return res[0].get("generated_text", "No response") if isinstance(res, list) else res.get("generated_text", "Error")
            return f"AI Error ({response.status_code}): {response.text}"
        except Exception as e:
            return f"Connection Failed: {str(e)}"

# --- MCP TOOLS ---
@mcp.tool()
async def audit_postpartum_gap(fhir_json: str) -> str:
    """Analyzes FHIR data for Fourth Trimester maternal care gaps."""
    # This is the prompt we optimized earlier
    prompt = f"Act as a Maternal Health Equity Agent. Analyze this FHIR bundle for a 12-week postpartum care gap in a high-risk patient: {fhir_json}"
    
    intelligence = MaternalHealthIntelligence()
    return intelligence.query_gemma(prompt)

# --- MOUNTING & HEALTH ---
app.mount("/mcp", mcp)

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}