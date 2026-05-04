import os
from fastapi import FastAPI
from fastmcp import FastMCP, Context
from neuro_logic import NeuroGuardIntelligence # Or paste the class here

app = FastAPI(title="NeuroGuard A2A Specialist")
# Use the FastMCP SDK to handle the protocol logic
mcp = FastMCP("NeuroGuard", description="Stroke & Vascular Intelligence")

# Define your tools (Ideas 1-7)
@mcp.tool()
async def full_vascular_audit(ctx: Context, age: float, hypertension: int, 
                             heart_disease: int, ever_married: str, 
                             work_type: str, avg_glucose_level: float, 
                             bmi: float, stroke: int) -> str:
    """Production-grade audit covering triage, rehab, and geriatric risk."""
    # SHARP headers are accessed via ctx.request_context in Prompt Opinion
    patient_id = ctx.request_context.get("x-sharp-patient-id", "GUEST")
    
    intel = NeuroGuardIntelligence()
    data = locals() # Captures the parameters
    
    # Run logic from Ideas 3, 5, 6, 7
    audit = intel.idea_3_comorbidity_orchestrator(data)
    rehab = intel.idea_5_post_stroke_navigator(data)
    
    return f"Audit for Patient {patient_id}: Risk Weight {audit['risk_weight']}. Rehab: {rehab['occupational_focus']}."

# A2A Discovery Endpoint
@app.get("/.well-known/agent.json")
async def get_agent_card():
    return {
        "name": "NeuroGuard Specialist",
        "description": "Expert agent for stroke risk and vascular health coordination.",
        "endpoint": f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME')}/mcp",
        "capabilities": {"inter_agent_chat": True, "context_propagation": "SHARP"},
        "skills": ["Stroke Triage", "SDoH Auditing", "Risk Simulation"]
    }

app.mount("/mcp", mcp.fastapi_app())