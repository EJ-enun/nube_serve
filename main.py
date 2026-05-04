import os
from fastapi import FastAPI
from fastmcp import FastMCP, Context
from typing import Dict, Any, List
from enum import Enum

app = FastAPI(title="NeuroGuard A2A Specialist")
mcp = FastMCP("NeuroGuard")

class RiskCategory(Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"
    CRITICAL = "Critical"

class NeuroGuardIntelligence:
    """Production-grade logic engine for all 7 Hackathon Ideas."""

    @staticmethod
    def idea_1_calculate_base_risk(data: Dict[str, Any]) -> float:
        score = (data.get('age', 0) * 0.1)
        if data.get('hypertension') == 1: score += 15
        if data.get('heart_disease') == 1: score += 20
        if data.get('avg_glucose_level', 0) > 150: score += 10
        return min(round(score, 2), 100.0)

    @staticmethod
    def idea_2_analyze_sdoh(data: Dict[str, Any]) -> str:
        res = data.get('Residence_type', 'Urban')
        work = data.get('work_type', 'Private')
        if res == "Rural" and work == "Self-employed":
            return "High Risk: Limited access to urgent care + high occupational stress."
        return "Standard environmental risk profile."

    @staticmethod
    def idea_3_comorbidity_orchestrator(data: Dict[str, Any]) -> Dict[str, Any]:
        score = sum([1 for condition in ['hypertension', 'heart_disease'] if data.get(condition) == 1])
        support_multiplier = 0.8 if data.get('ever_married') == "Yes" else 1.2
        return {
            "comorbidity_score": score,
            "social_support_factor": "Strong" if support_multiplier < 1 else "Limited",
            "risk_weight": round(score * 2.5 * support_multiplier, 2)
        }

    @staticmethod
    def idea_4_lifestyle_pivot_sim(current_bmi: float, target_bmi: float, smokes: str) -> str:
        improvement = (current_bmi - target_bmi) * 1.5
        if smokes == "smokes": improvement += 25
        return f"Risk Reduction: {round(improvement, 1)}% decrease in 5-year stroke probability."

    @staticmethod
    def idea_5_post_stroke_navigator(data: Dict[str, Any]) -> Dict[str, Any]:
        work_type = data.get('work_type', 'Private')
        focus_area = "Manual Dexterity & Mobility" if work_type in ["Self-employed", "Private"] else "Cognitive Pacing"
        return {
            "rehab_status": "Active" if data.get('stroke') == 1 else "Preventative",
            "occupational_focus": focus_area
        }

    @staticmethod
    def idea_6_silent_risk_monitor(data: Dict[str, Any]) -> Dict[str, Any]:
        glucose = data.get('avg_glucose_level', 0)
        is_silent = glucose > 180 and data.get('hypertension') == 0 and data.get('heart_disease') == 0
        return {
            "metabolic_flag": "Elevated" if glucose > 140 else "Normal",
            "silent_risk_detected": is_silent
        }

    @staticmethod
    def idea_7_geriatric_shield(data: Dict[str, Any]) -> Dict[str, Any]:
        age = data.get('age', 0)
        weight = 2.0 if age > 75 else 1.5 if age >= 65 else 1.0
        return {
            "shield_active": age >= 65,
            "age_adjusted_weight": weight
        }

# --- MCP TOOLS EXPPOSED TO THE AGENT ---

@mcp.tool()
async def evaluate_full_vascular_profile(ctx: Context, age: float, hypertension: int, heart_disease: int, 
                                        ever_married: str, work_type: str, avg_glucose_level: float, 
                                        bmi: float, stroke: int, residence_type: str) -> str:
    """The master tool. Combines Ideas 1, 2, 3, 5, 6, and 7 into a single clinical audit."""
    
    patient_id = ctx.request_context.get("x-sharp-patient-id", "GUEST_PATIENT")
    
    data = {
        "age": age, "hypertension": hypertension, "heart_disease": heart_disease,
        "ever_married": ever_married, "work_type": work_type, "Residence_type": residence_type,
        "avg_glucose_level": avg_glucose_level, "bmi": bmi, "stroke": stroke
    }

    intel = NeuroGuardIntelligence()
    
    triage_risk = intel.idea_1_calculate_base_risk(data)
    sdoh = intel.idea_2_analyze_sdoh(data)
    comorbidity = intel.idea_3_comorbidity_orchestrator(data)
    rehab = intel.idea_5_post_stroke_navigator(data)
    silent_risk = intel.idea_6_silent_risk_monitor(data)
    geriatric = intel.idea_7_geriatric_shield(data)

    return f"""
    --- NEUROGUARD AUDIT FOR PATIENT: {patient_id} ---
    1. Base Triage Risk: {triage_risk}%
    2. SDoH Analysis: {sdoh}
    3. Comorbidity Weight: {comorbidity['risk_weight']} (Support: {comorbidity['social_support_factor']})
    4. Silent Risk (Metabolic): {'DETECTED' if silent_risk['silent_risk_detected'] else 'Clear'}
    5. Geriatric Shield: {'ACTIVE' if geriatric['shield_active'] else 'N/A'}
    6. Care Plan Focus: {rehab['occupational_focus']}
    """

@mcp.tool()
async def simulate_lifestyle_changes(ctx: Context, current_bmi: float, target_bmi: float, smoking_status: str) -> str:
    """Idea 4: Run a 'What-If' simulation for patient motivation."""
    intel = NeuroGuardIntelligence()
    return intel.idea_4_lifestyle_pivot_sim(current_bmi, target_bmi, smoking_status)

# --- A2A DISCOVERY ENDPOINT ---

@app.get("/.well-known/agent.json")
async def get_agent_card():
    # Render automatically sets RENDER_EXTERNAL_HOSTNAME, but we fallback to localhost for testing
    host = os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'localhost:8000')
    protocol = "https" if "render" in host else "http"
    
    return {
        "name": "NeuroGuard Specialist",
        "description": "Expert agent for stroke risk and vascular health coordination.",
        "endpoint": f"{protocol}://{host}/mcp",
        "capabilities": {"inter_agent_chat": True, "context_propagation": "SHARP"},
        "skills": ["Stroke Triage", "SDoH Auditing", "Risk Simulation", "Geriatric Assessment"]
    }

print(dir(mcp)[-10:])          # show recently added attributes
app.mount("/mcp", mcp.app)