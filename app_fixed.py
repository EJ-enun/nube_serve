
import os
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP

# =============================================================================
# MCP SERVER CONFIGURATION
# =============================================================================

mcp = FastMCP(
    "FourthTrimesterCareAgent",
    instructions="""
    PromptOpinion FHIR-context compatible server.

    This server is FHIR-aware and designed for maternal healthcare workflows.

    Supported standards:
    - HL7 FHIR R4
    - MCP protocol

    Supported contexts:
    - patient
    - encounter
    - practitioner

    Supported workflows:
    - postpartum risk review
    - maternal care gap detection
    - equity-focused prioritization
    - clinical outreach support

    This server supports:
    - MCP tools
    - MCP resources
    - FHIR capability discovery
    - structured clinical context
    """
)

_original_get_capabilities = mcp._mcp_server.get_capabilities


def _patched_get_capabilities(notification_options, experimental_capabilities):
    caps = _original_get_capabilities(notification_options, experimental_capabilities)
    extras = getattr(caps, "model_extra", None)
    if extras is None:
        caps.model_extra = {}
        extras = caps.model_extra

    extras["extensions"] = {
        "ai.promptopinion/fhir-context": {
            "scopes": [
                {"name": "patient/Patient.rs", "required": True},
                {"name": "patient/Observation.rs"},
                {"name": "patient/Condition.rs"},
                {"name": "patient/Encounter.rs"},
                {"name": "patient/Appointment.rs"},
                {"name": "patient/ServiceRequest.rs"},
                {"name": "patient/Communication.rs"},
                {"name": "patient/Task.rs"},
                {"name": "patient/CarePlan.rs"},
            ]
        }
    }
    return caps


mcp._mcp_server.get_capabilities = _patched_get_capabilities

# IMPORTANT:
# path="/" + app.mount("/mcp", ...)
# results in MCP endpoint at /mcp
mcp_app = mcp.http_app(path="/")

app = FastAPI(
    title="Maternal Health Equity Agent",
    lifespan=mcp_app.lifespan,
)

# =============================================================================
# POLICY INSTRUCTIONS
# =============================================================================

POLICY_INSTRUCTIONS = """
MISSION:
Identify 0-12 week postpartum care gaps to prevent maternal mortality.

DEFINITION:
A gap exists if a delivery is documented but no follow-up visit is found.

RISK MARKERS:
- Preeclampsia
- Hypertension
- Gestational Diabetes
- C-Section
- Severe-range blood pressure
- Missing postpartum depression screening

EQUITY POLICY:
Elevate priority for Black/African American patients due to higher morbidity risks.

TASK:
Analyze the FHIR bundle.

OUTPUT FORMAT:
Return structured JSON:
{
  "priority_level": "URGENT | HIGH | MEDIUM",
  "clinical_reasoning": "Reason based on markers like BP, C-section, or missing screening",
  "equity_context": "Reasoning for race-based priority elevation",
  "outreach_message": "Clinical, brief, action-focused scheduling instruction"
}
"""

# =============================================================================
# CONSTANTS
# =============================================================================

POSTPARTUM_WINDOW_DAYS = 84  # 12 weeks
SEVERE_SBP = 160
SEVERE_DBP = 110

SDOH_TERMS = [
    "food insecurity",
    "housing instability",
    "transportation",
    "transport",
    "childcare",
    "language",
    "limited english",
    "financial strain",
    "utility insecurity",
    "internet access",
]

POSTPARTUM_DEPRESSION_TERMS = [
    "epds",
    "phq-9",
    "phq9",
    "postpartum depression",
    "depression screening",
    "mood screening",
    "edinburgh postnatal depression scale",
]

# =============================================================================
# HELPERS
# =============================================================================


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_now_iso() -> str:
    return _utc_now().isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex[:12]}"


def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _resource_entries(bundle: dict[str, Any], resource_type: str) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for entry in bundle.get("entry", []) or []:
        resource = entry.get("resource", {})
        if resource.get("resourceType") == resource_type:
            entries.append(resource)
    return entries


def _resource_subject_matches(resource: dict[str, Any], patient_id: str) -> bool:
    subject = resource.get("subject", {})
    ref = subject.get("reference", "")
    if isinstance(ref, str) and ref.endswith(patient_id):
        return True
    # Some bundle records may reference Patient/{id}
    if ref == f"Patient/{patient_id}":
        return True
    return False


def _get_patient(bundle: dict[str, Any], patient_id: str) -> dict[str, Any]:
    for patient in _resource_entries(bundle, "Patient"):
        if patient.get("id") == patient_id:
            return patient
    return {}


def _patient_has_black_race(patient: dict[str, Any]) -> bool:
    for ext in patient.get("extension", []) or []:
        for nested in ext.get("extension", []) or []:
            coding = nested.get("valueCoding", {})
            code = str(coding.get("code", "")).strip()
            display = str(coding.get("display", "")).lower()
            if code == "2054-5" or "black" in display:
                return True
    return False


def _obs_text(obs: dict[str, Any]) -> str:
    text_bits = []
    for coding in obs.get("code", {}).get("coding", []) or []:
        text_bits.append(str(coding.get("display", "")))
        text_bits.append(str(coding.get("code", "")))
    text_bits.append(str(obs.get("code", {}).get("text", "")))
    text_bits.append(str(obs.get("valueString", "")))
    text_bits.append(str(obs.get("interpretation", [{}])[0].get("text", "")) if isinstance(obs.get("interpretation"), list) and obs.get("interpretation") else "")
    return " ".join(t for t in text_bits if t).lower()


def _detect_delivery_event(bundle: dict[str, Any], patient_id: str) -> tuple[Optional[datetime], Optional[str], Optional[str]]:
    """Return (delivery_dt, resource_id, resource_type)."""
    candidates: list[tuple[datetime, str, str]] = []

    for encounter in _resource_entries(bundle, "Encounter"):
        if not _resource_subject_matches(encounter, patient_id):
            continue
        period = encounter.get("period", {})
        start = _parse_dt(period.get("start"))
        if start:
            # Prefer obstetric or delivery-like encounters, but accept any delivery encounter.
            text = " ".join([
                str(encounter.get("type", "")),
                str(encounter.get("reasonCode", "")),
                str(encounter.get("class", {}).get("display", "")),
            ]).lower()
            if any(term in text for term in ["delivery", "obstetric", "labor", "postpartum", "inpatient"]) or True:
                candidates.append((start, encounter.get("id", ""), "Encounter"))

    for procedure in _resource_entries(bundle, "Procedure"):
        if not _resource_subject_matches(procedure, patient_id):
            continue
        code_text = " ".join(
            str(x.get("display", "")).lower() for x in procedure.get("code", {}).get("coding", []) or []
        )
        code_text += " " + str(procedure.get("code", {}).get("text", "")).lower()
        if any(term in code_text for term in ["cesarean", "c-section", "c section", "delivery", "birth", "obstetric"]):
            performed = _parse_dt(procedure.get("performedDateTime"))
            if performed:
                candidates.append((performed, procedure.get("id", ""), "Procedure"))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def _detect_follow_up(bundle: dict[str, Any], patient_id: str, delivery_dt: Optional[datetime]) -> tuple[Optional[datetime], Optional[str], Optional[str]]:
    """Return (followup_dt, resource_id, resource_type)."""
    if not delivery_dt:
        return None, None, None

    window_end = delivery_dt + timedelta(days=POSTPARTUM_WINDOW_DAYS)
    candidates: list[tuple[datetime, str, str]] = []

    for appt in _resource_entries(bundle, "Appointment"):
        if not any(
            (part.get("actor", {}).get("reference", "") or "").endswith(patient_id)
            for part in appt.get("participant", []) or []
        ):
            continue

        status = str(appt.get("status", "")).lower()
        text_blob = " ".join([
            str(appt.get("description", "")),
            str(appt.get("comment", "")),
            str(appt.get("reasonCode", "")),
        ]).lower()

        if any(term in text_blob for term in ["postpartum", "post-partum", "6-week", "12-week", "follow-up"]):
            start = _parse_dt(appt.get("start"))
            if start and delivery_dt <= start <= window_end and status in {"booked", "fulfilled", "arrived", "completed", "proposed"}:
                candidates.append((start, appt.get("id", ""), "Appointment"))

    for enc in _resource_entries(bundle, "Encounter"):
        if not _resource_subject_matches(enc, patient_id):
            continue
        status = str(enc.get("status", "")).lower()
        text_blob = " ".join([
            str(enc.get("type", "")),
            str(enc.get("reasonCode", "")),
            str(enc.get("class", {}).get("display", "")),
        ]).lower()

        if any(term in text_blob for term in ["postpartum", "post-partum", "follow-up"]):
            start = _parse_dt(enc.get("period", {}).get("start"))
            if start and delivery_dt <= start <= window_end and status in {"finished", "completed", "arrived"}:
                candidates.append((start, enc.get("id", ""), "Encounter"))

    if not candidates:
        return None, None, None

    candidates.sort(key=lambda x: x[0])
    return candidates[0]


def _detect_severe_bp(bundle: dict[str, Any], patient_id: str) -> bool:
    for obs in _resource_entries(bundle, "Observation"):
        if not _resource_subject_matches(obs, patient_id):
            continue

        code_blob = _obs_text(obs)
        is_bp = any(term in code_blob for term in ["blood pressure", "systolic", "diastolic", "85354-9", "8480-6", "8462-4"])
        if not is_bp:
            continue

        for comp in obs.get("component", []) or []:
            comp_code = " ".join(
                str(x.get("display", "")).lower() for x in comp.get("code", {}).get("coding", []) or []
            )
            comp_code += " " + str(comp.get("code", {}).get("text", "")).lower()
            value = comp.get("valueQuantity", {}).get("value")
            if value is None:
                continue
            try:
                value = float(value)
            except Exception:
                continue

            if ("systolic" in comp_code or "8480-6" in comp_code) and value >= SEVERE_SBP:
                return True
            if ("diastolic" in comp_code or "8462-4" in comp_code) and value >= SEVERE_DBP:
                return True

    return False


def _detect_risk_markers(bundle: dict[str, Any], patient_id: str) -> dict[str, Any]:
    flags = {
        "preeclampsia": False,
        "hypertension": False,
        "gestational_diabetes": False,
        "c_section": False,
        "severe_bp": _detect_severe_bp(bundle, patient_id),
        "black_patient": False,
        "sdoh_flags": [],
        "depression_screening_missing": False,
        "depression_screening_found": False,
        "depression_screening_positive": False,
    }

    patient = _get_patient(bundle, patient_id)
    flags["black_patient"] = _patient_has_black_race(patient)

    for cond in _resource_entries(bundle, "Condition"):
        if not _resource_subject_matches(cond, patient_id):
            continue

        cond_text = " ".join(
            str(x.get("display", "")).lower() for x in cond.get("code", {}).get("coding", []) or []
        )
        cond_text += " " + str(cond.get("code", {}).get("text", "")).lower()

        if "preeclampsia" in cond_text:
            flags["preeclampsia"] = True
        if "hypertens" in cond_text:
            flags["hypertension"] = True
        if "gestational diabetes" in cond_text or "gdm" in cond_text:
            flags["gestational_diabetes"] = True

    for proc in _resource_entries(bundle, "Procedure"):
        if not _resource_subject_matches(proc, patient_id):
            continue

        proc_text = " ".join(
            str(x.get("display", "")).lower() for x in proc.get("code", {}).get("coding", []) or []
        )
        proc_text += " " + str(proc.get("code", {}).get("text", "")).lower()

        if any(term in proc_text for term in ["cesarean", "c-section", "c section"]):
            flags["c_section"] = True

    for obs in _resource_entries(bundle, "Observation"):
        if not _resource_subject_matches(obs, patient_id):
            continue

        blob = _obs_text(obs)

        # SDOH
        for term in SDOH_TERMS:
            if term in blob and term not in flags["sdoh_flags"]:
                flags["sdoh_flags"].append(term)

        # Postpartum depression screening
        if any(term in blob for term in POSTPARTUM_DEPRESSION_TERMS):
            flags["depression_screening_found"] = True
            # Positive / elevated result detection, broadly interpreted.
            score = obs.get("valueQuantity", {}).get("value")
            if score is not None:
                try:
                    score_f = float(score)
                    if score_f >= 10:
                        flags["depression_screening_positive"] = True
                except Exception:
                    pass
            text_value = str(obs.get("valueString", "")).lower()
            if any(term in text_value for term in ["positive", "high", "elevated", "severe"]):
                flags["depression_screening_positive"] = True

    flags["depression_screening_missing"] = not flags["depression_screening_found"]
    return flags


def _priority_from_flags(flags: dict[str, Any], followup_missing: bool) -> str:
    score = 0

    if followup_missing:
        score += 3
    if flags["preeclampsia"] or flags["severe_bp"]:
        score += 4
    if flags["hypertension"]:
        score += 2
    if flags["gestational_diabetes"]:
        score += 1
    if flags["c_section"]:
        score += 1
    if flags["depression_screening_missing"]:
        score += 1
    if flags["depression_screening_positive"]:
        score += 2
    if flags["black_patient"]:
        score += 1
    if flags["sdoh_flags"]:
        score += 1

    if (flags["preeclampsia"] or flags["severe_bp"] or flags["depression_screening_positive"]) and followup_missing:
        return "URGENT"
    if score >= 5:
        return "HIGH"
    return "MEDIUM"


def _build_clinical_reasoning(
    delivery_dt: Optional[datetime],
    delivery_id: Optional[str],
    delivery_type: Optional[str],
    followup_dt: Optional[datetime],
    followup_id: Optional[str],
    followup_type: Optional[str],
    flags: dict[str, Any],
) -> str:
    parts: list[str] = []
    if delivery_dt:
        parts.append(
            f"Delivery was documented on {delivery_dt.date().isoformat()}"
            + (f" ({delivery_type} {delivery_id})" if delivery_id else "")
        )
    else:
        parts.append("No delivery event could be confirmed in the submitted FHIR data")

    if followup_dt:
        parts.append(
            f"follow-up was found on {followup_dt.date().isoformat()}"
            + (f" ({followup_type} {followup_id})" if followup_id else "")
        )
    else:
        parts.append("no postpartum follow-up was found within the 12-week window")

    risks = []
    if flags["preeclampsia"]:
        risks.append("preeclampsia")
    if flags["hypertension"]:
        risks.append("hypertension")
    if flags["gestational_diabetes"]:
        risks.append("gestational diabetes")
    if flags["c_section"]:
        risks.append("C-section recovery")
    if flags["severe_bp"]:
        risks.append("severe-range blood pressure")
    if flags["depression_screening_missing"]:
        risks.append("missing postpartum depression screening")
    if flags["depression_screening_positive"]:
        risks.append("positive postpartum depression screening")

    if risks:
        parts.append("Risk markers identified: " + ", ".join(risks) + ".")
    else:
        parts.append("No major high-risk marker was documented.")

    if flags["sdoh_flags"]:
        parts.append("Social risk indicators noted: " + ", ".join(flags["sdoh_flags"]) + ".")

    return " ".join(parts)


def _build_equity_context(flags: dict[str, Any]) -> str:
    if flags["black_patient"]:
        return (
            "Black maternal patients face a disproportionate burden of preventable postpartum morbidity and mortality. "
            "Closing the follow-up gap supports earlier detection, treatment, and care coordination."
        )
    return (
        "Closing postpartum follow-up gaps reduces the chance that complications, including blood pressure issues "
        "and postpartum depression, are missed during the highest-risk recovery period."
    )


def _build_outreach_message(patient_id: str, priority: str, flags: dict[str, Any]) -> str:
    message = (
        f"Patient {patient_id}, our records show your postpartum follow-up visit is not yet documented. "
        f"Please schedule your 6-week or 12-week postpartum appointment as soon as possible."
    )
    if priority == "URGENT":
        message += " Because of your risk factors, please contact the clinic today."
    if flags["depression_screening_missing"]:
        message += " We will also complete postpartum depression screening at the visit."
    return message


def _find_matching_postpartum_followup(bundle: dict[str, Any], patient_id: str, delivery_dt: Optional[datetime]) -> tuple[Optional[datetime], Optional[str], Optional[str]]:
    return _detect_follow_up(bundle, patient_id, delivery_dt)


def _maybe_add_patient_race_extension(patient: dict[str, Any]) -> None:
    # No-op helper retained for future extension point.
    return


def _fhir_resource_ref(resource_type: str, resource_id: str) -> dict[str, str]:
    return {"reference": f"{resource_type}/{resource_id}"}


async def _persist_fhir_resource(resource: dict[str, Any]) -> dict[str, Any]:
    """POST to configured FHIR server if available, otherwise simulate."""
    base_url = os.environ.get("FHIR_BASE_URL")
    token = os.environ.get("FHIR_TOKEN")

    if not base_url:
        return {
            "persisted": False,
            "mode": "simulated",
            "resource": resource,
        }

    headers = {"Content-Type": "application/fhir+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    url = f"{base_url.rstrip('/')}/{resource['resourceType']}"
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(url, json=resource, headers=headers)
        response.raise_for_status()
        return {
            "persisted": True,
            "mode": "fhir_server",
            "status_code": response.status_code,
            "response": response.json() if response.content else {},
        }


# =============================================================================
# CORE MCP TOOL
# =============================================================================


@mcp.tool()
async def audit_postpartum_gap(
    patient_id: str,
    fhir_bundle: dict[str, Any],
    encounter_id: Optional[str] = None,
    practitioner_id: Optional[str] = None,
) -> dict[str, Any]:
    """Analyze HL7 FHIR R4 postpartum data for maternal care gaps."""
    if not isinstance(fhir_bundle, dict):
        raise HTTPException(status_code=400, detail="fhir_bundle must be a JSON object")

    delivery_dt, delivery_id, delivery_type = _detect_delivery_event(fhir_bundle, patient_id)
    followup_dt, followup_id, followup_type = _find_matching_postpartum_followup(fhir_bundle, patient_id, delivery_dt)
    flags = _detect_risk_markers(fhir_bundle, patient_id)

    if delivery_dt is None:
        return {
            "priority_level": "MEDIUM",
            "clinical_reasoning": "No delivery event could be confirmed in the provided FHIR data.",
            "equity_context": _build_equity_context(flags),
            "outreach_message": "Please review the chart and confirm whether postpartum follow-up is needed.",
            "patient_id": patient_id,
            "encounter_id": encounter_id,
            "practitioner_id": practitioner_id,
            "gap_status": "UNKNOWN",
        }

    followup_missing = followup_dt is None
    priority_level = _priority_from_flags(flags, followup_missing)
    clinical_reasoning = _build_clinical_reasoning(
        delivery_dt,
        delivery_id,
        delivery_type,
        followup_dt,
        followup_id,
        followup_type,
        flags,
    )
    equity_context = _build_equity_context(flags)
    outreach_message = _build_outreach_message(patient_id, priority_level, flags)

    return {
        "priority_level": priority_level,
        "clinical_reasoning": clinical_reasoning,
        "equity_context": equity_context,
        "outreach_message": outreach_message,
        "patient_id": patient_id,
        "encounter_id": encounter_id,
        "practitioner_id": practitioner_id,
        "gap_status": "OPEN" if followup_missing else "CLOSED",
        "delivery_date": delivery_dt.isoformat() if delivery_dt else None,
        "follow_up_date": followup_dt.isoformat() if followup_dt else None,
        "risk_flags": flags,
    }


# =============================================================================
# NEW MCP TOOLS
# =============================================================================


@mcp.tool()
async def schedule_postpartum_visit(
    patient_id: str,
    visit_type: str = "6-week",
    preferred_start: Optional[str] = None,
    preferred_end: Optional[str] = None,
    practitioner_id: Optional[str] = None,
    fhir_bundle: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a ServiceRequest and Appointment proposal for postpartum follow-up."""
    visit_label_map = {
        "6-week": "6-week postpartum visit",
        "12-week": "12-week postpartum visit",
        "comprehensive": "12-week comprehensive postpartum follow-up",
    }
    visit_label = visit_label_map.get(visit_type.lower(), f"{visit_type} postpartum visit")

    equity_notes: list[str] = []
    if fhir_bundle:
        patient = _get_patient(fhir_bundle, patient_id)
        if _patient_has_black_race(patient):
            equity_notes.append("Black/African American patient")
        flags = _detect_risk_markers(fhir_bundle, patient_id)
        if any(term in flags["sdoh_flags"] for term in ["transportation", "transport"]):
            equity_notes.append("transportation difficulty")
        if any(term in flags["sdoh_flags"] for term in ["language", "limited english"]):
            equity_notes.append("limited English proficiency")
        if any(term in flags["sdoh_flags"] for term in ["childcare", "financial strain"]):
            equity_notes.append("scheduling barrier / work-hour constraint")

    service_request_id = _new_id("sr")
    appointment_id = _new_id("appt")

    service_request: dict[str, Any] = {
        "resourceType": "ServiceRequest",
        "id": service_request_id,
        "status": "active",
        "intent": "order",
        "subject": _fhir_resource_ref("Patient", patient_id),
        "code": {"text": visit_label},
        "occurrencePeriod": {
            "start": preferred_start,
            "end": preferred_end,
        },
        "note": [
            {"text": f"Postpartum follow-up scheduling request created for {visit_label}."}
        ],
    }

    appointment: dict[str, Any] = {
        "resourceType": "Appointment",
        "id": appointment_id,
        "status": "proposed",
        "description": visit_label,
        "participant": [
            {
                "actor": _fhir_resource_ref("Patient", patient_id),
                "status": "accepted",
            }
        ],
        "basedOn": [
            _fhir_resource_ref("ServiceRequest", service_request_id)
        ],
        "requestedPeriod": [],
        "comment": "Offer telehealth or extended-hours slot if needed.",
    }

    if practitioner_id:
        appointment["participant"].append(
            {
                "actor": _fhir_resource_ref("Practitioner", practitioner_id),
                "status": "needs-action",
            }
        )

    if preferred_start and preferred_end:
        appointment["requestedPeriod"] = [{"start": preferred_start, "end": preferred_end}]
    elif preferred_start:
        appointment["requestedPeriod"] = [{"start": preferred_start}]

    if equity_notes:
        appointment["comment"] = (
            "Offer telehealth or extended-hours slot if needed. "
            f"Equity/access flags: {', '.join(equity_notes)}."
        )

    sr_result = await _persist_fhir_resource(service_request)
    appt_result = await _persist_fhir_resource(appointment)

    return {
        "patient_id": patient_id,
        "visit_type": visit_type,
        "service_request_id": service_request_id,
        "appointment_id": appointment_id,
        "confirmation_message": (
            f"Postpartum visit request created for {visit_label}. "
            "Please finalize scheduling with the patient and confirm the appointment."
        ),
        "equity_notes": equity_notes,
        "service_request": service_request if not sr_result.get("persisted") else sr_result,
        "appointment": appointment if not appt_result.get("persisted") else appt_result,
    }


@mcp.tool()
async def send_outreach_message(
    patient_id: str,
    message_content: str,
    channel: str,
    appointment_reference: Optional[str] = None,
) -> dict[str, Any]:
    """Create a Communication resource and mark the outreach as sent."""
    communication_id = _new_id("comm")

    communication: dict[str, Any] = {
        "resourceType": "Communication",
        "id": communication_id,
        "status": "completed",
        "sent": _utc_now_iso(),
        "subject": _fhir_resource_ref("Patient", patient_id),
        "payload": [
            {"contentString": message_content}
        ],
        "medium": [
            {"text": channel}
        ],
    }

    if appointment_reference:
        communication["basedOn"] = [{"reference": appointment_reference}]

    comm_result = await _persist_fhir_resource(communication)

    return {
        "patient_id": patient_id,
        "communication_id": communication_id,
        "channel": channel,
        "delivery_receipt": {
            "status": "sent",
            "channel": channel,
            "timestamp": _utc_now_iso(),
            "simulated": not comm_result.get("persisted", False),
        },
        "communication": communication if not comm_result.get("persisted") else comm_result,
    }


@mcp.tool()
async def verify_follow_up_completion(
    patient_id: str,
    fhir_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Check whether postpartum follow-up is documented within the 0-12 week window."""
    if not isinstance(fhir_bundle, dict):
        raise HTTPException(status_code=400, detail="fhir_bundle must be a JSON object")

    delivery_dt, delivery_id, delivery_type = _detect_delivery_event(fhir_bundle, patient_id)
    follow_dt, follow_id, follow_type = _find_matching_postpartum_followup(fhir_bundle, patient_id, delivery_dt)

    if follow_dt:
        return {
            "gap_status": "CLOSED",
            "patient_id": patient_id,
            "delivery_date": delivery_dt.isoformat() if delivery_dt else None,
            "delivery_id": delivery_id,
            "delivery_type": delivery_type,
            "follow_up_date": follow_dt.isoformat(),
            "follow_up_id": follow_id,
            "follow_up_type": follow_type,
        }

    return {
        "gap_status": "OPEN",
        "patient_id": patient_id,
        "delivery_date": delivery_dt.isoformat() if delivery_dt else None,
        "delivery_id": delivery_id,
        "delivery_type": delivery_type,
        "follow_up_date": None,
        "follow_up_id": None,
        "follow_up_type": None,
        "note": "No documented postpartum follow-up found within the 0-12 week window.",
    }


@mcp.tool()
async def escalate_to_clinician(
    patient_id: str,
    reason_code: str,
    clinician_reference: Optional[str] = None,
    fhir_bundle: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Create a Task for urgent clinician review when risk is urgent."""
    urgent = False
    if fhir_bundle:
        flags = _detect_risk_markers(fhir_bundle, patient_id)
        urgent = bool(flags["severe_bp"] or flags["preeclampsia"] or flags["depression_screening_positive"])

    if not urgent and "severe" in reason_code.lower():
        urgent = True

    if not urgent:
        return {
            "task_created": False,
            "patient_id": patient_id,
            "message": "Clinical escalation not required based on current criteria.",
        }

    task_id = _new_id("task")
    task: dict[str, Any] = {
        "resourceType": "Task",
        "id": task_id,
        "status": "requested",
        "intent": "order",
        "priority": "urgent",
        "for": _fhir_resource_ref("Patient", patient_id),
        "authoredOn": _utc_now_iso(),
        "description": f"Urgent postpartum review: {reason_code}. Call patient within 2 hours.",
        "code": {"text": reason_code},
        "note": [
            {"text": "Immediate human review requested by maternal safety workflow."}
        ],
    }

    if clinician_reference:
        task["owner"] = {"reference": clinician_reference}

    task_result = await _persist_fhir_resource(task)

    return {
        "task_created": True,
        "task_id": task_id,
        "patient_id": patient_id,
        "recommended_action": "Call patient within 2 hours.",
        "task": task if not task_result.get("persisted") else task_result,
    }


@mcp.tool()
async def generate_care_plan_summary(
    patient_id: str,
    fhir_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Create a postpartum CarePlan and patient-facing summary."""
    if not isinstance(fhir_bundle, dict):
        raise HTTPException(status_code=400, detail="fhir_bundle must be a JSON object")

    flags = _detect_risk_markers(fhir_bundle, patient_id)
    delivery_dt, delivery_id, _ = _detect_delivery_event(fhir_bundle, patient_id)

    careplan_id = _new_id("cp")
    activities: list[dict[str, Any]] = [
        {
            "detail": {
                "kind": "Appointment",
                "code": {"text": "Postpartum follow-up visit"},
                "description": "Schedule postpartum follow-up",
            }
        }
    ]

    if flags["severe_bp"] or flags["hypertension"] or flags["preeclampsia"]:
        activities.append(
            {
                "detail": {
                    "kind": "Observation",
                    "code": {"text": "Blood pressure monitoring"},
                    "description": "Home or clinic blood pressure checks as directed",
                }
            }
        )

    if flags["gestational_diabetes"]:
        activities.append(
            {
                "detail": {
                    "kind": "ReferralRequest",
                    "code": {"text": "Diabetes follow-up"},
                    "description": "Follow-up for glucose monitoring / primary care",
                }
            }
        )

    if flags["depression_screening_missing"] or flags["depression_screening_positive"]:
        activities.append(
            {
                "detail": {
                    "kind": "Observation",
                    "code": {"text": "Postpartum depression screening"},
                    "description": "Complete EPDS or PHQ-9 screening and review results",
                }
            }
        )
        if flags["depression_screening_positive"]:
            activities.append(
                {
                    "detail": {
                        "kind": "ReferralRequest",
                        "code": {"text": "Behavioral health referral"},
                        "description": "Urgent behavioral health follow-up for positive screening",
                    }
                }
            )

    if flags["sdoh_flags"]:
        activities.append(
            {
                "detail": {
                    "kind": "ReferralRequest",
                    "code": {"text": "Social needs referral"},
                    "description": "Refer for transportation, food, childcare, or language support",
                }
            }
        )

    care_plan: dict[str, Any] = {
        "resourceType": "CarePlan",
        "id": careplan_id,
        "status": "active",
        "intent": "plan",
        "subject": _fhir_resource_ref("Patient", patient_id),
        "period": {
            "start": delivery_dt.isoformat() if delivery_dt else _utc_now_iso(),
            "end": (delivery_dt + timedelta(days=POSTPARTUM_WINDOW_DAYS)).isoformat() if delivery_dt else None,
        },
        "activity": activities,
        "note": [
            {"text": "Postpartum care plan generated from delivery, risk, and equity review."}
        ],
    }

    careplan_result = await _persist_fhir_resource(care_plan)

    patient_summary = ["Schedule postpartum follow-up as soon as possible."]
    if flags["severe_bp"] or flags["preeclampsia"] or flags["hypertension"]:
        patient_summary.append("Monitor blood pressure closely.")
    if flags["gestational_diabetes"]:
        patient_summary.append("Continue glucose-related follow-up.")
    if flags["c_section"]:
        patient_summary.append("Follow C-section recovery instructions.")
    if flags["depression_screening_missing"]:
        patient_summary.append("Complete postpartum depression screening at the visit.")
    if flags["depression_screening_positive"]:
        patient_summary.append("Behavioral health follow-up is recommended because screening was positive.")
    if flags["sdoh_flags"]:
        patient_summary.append("Address social needs that may affect follow-up.")
    if flags["black_patient"]:
        patient_summary.append("Care coordination is especially important because postpartum risk is elevated.")

    return {
        "patient_id": patient_id,
        "careplan_id": careplan_id,
        "care_plan": care_plan if not careplan_result.get("persisted") else careplan_result,
        "patient_portal_summary": " ".join(patient_summary),
        "risk_flags": flags,
        "delivery_id": delivery_id,
    }


@mcp.tool()
async def analyze_sdoh_risks(
    patient_id: str,
    fhir_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Scan observations for social determinants of health and return actionable flags."""
    if not isinstance(fhir_bundle, dict):
        raise HTTPException(status_code=400, detail="fhir_bundle must be a JSON object")

    flags = _detect_risk_markers(fhir_bundle, patient_id).get("sdoh_flags", [])
    if not flags:
        return {
            "patient_id": patient_id,
            "sdoh_status": "MISSING_SCREENING",
            "flagged_risks": [],
            "recommended_action": "Administer standardized SDOH screening and document results.",
        }

    referral_map = {
        "transportation": "Transportation assistance referral",
        "transport": "Transportation assistance referral",
        "food insecurity": "WIC / food support referral",
        "housing instability": "Social work / housing referral",
        "childcare": "Care navigation / childcare support referral",
        "language": "Interpreter support / language services",
        "limited english": "Interpreter support / language services",
        "financial strain": "Social work / benefits screening",
        "utility insecurity": "Social work referral",
        "internet access": "Telehealth access support",
    }

    referrals = []
    for flag in flags:
        referral = referral_map.get(flag)
        if referral and referral not in referrals:
            referrals.append(referral)

    return {
        "patient_id": patient_id,
        "sdoh_status": "FLAGGED",
        "flagged_risks": flags,
        "suggested_referrals": referrals,
    }


@mcp.tool()
async def detect_postpartum_depression_screening_gap(
    patient_id: str,
    fhir_bundle: dict[str, Any],
) -> dict[str, Any]:
    """Detect whether postpartum depression screening is missing or positive."""
    if not isinstance(fhir_bundle, dict):
        raise HTTPException(status_code=400, detail="fhir_bundle must be a JSON object")

    delivery_dt, delivery_id, delivery_type = _detect_delivery_event(fhir_bundle, patient_id)
    flags = _detect_risk_markers(fhir_bundle, patient_id)

    if delivery_dt is None:
        return {
            "gap_status": "UNKNOWN",
            "patient_id": patient_id,
            "delivery_id": None,
            "delivery_type": None,
            "screening_found": flags["depression_screening_found"],
            "screening_positive": flags["depression_screening_positive"],
            "note": "No delivery event found; postpartum depression screening window cannot be evaluated.",
        }

    if flags["depression_screening_positive"]:
        return {
            "gap_status": "OPEN",
            "patient_id": patient_id,
            "delivery_id": delivery_id,
            "delivery_type": delivery_type,
            "screening_found": True,
            "screening_positive": True,
            "recommended_action": "Escalate to clinician and behavioral health.",
        }

    if flags["depression_screening_missing"]:
        return {
            "gap_status": "OPEN",
            "patient_id": patient_id,
            "delivery_id": delivery_id,
            "delivery_type": delivery_type,
            "screening_found": False,
            "screening_positive": False,
            "recommended_action": "Complete postpartum depression screening at the postpartum visit.",
        }

    return {
        "gap_status": "CLOSED",
        "patient_id": patient_id,
        "delivery_id": delivery_id,
        "delivery_type": delivery_type,
        "screening_found": True,
        "screening_positive": False,
        "recommended_action": "No action needed for screening gap; continue routine monitoring.",
    }


@mcp.tool()
async def build_outreach_queue(
    patients: list[dict[str, Any]],
) -> dict[str, Any]:
    """Rank multiple postpartum patients by urgency for care coordination."""
    ranked = []
    for item in patients:
        patient_id = item.get("patient_id")
        bundle = item.get("fhir_bundle", {})
        if not patient_id or not isinstance(bundle, dict):
            continue

        review = await audit_postpartum_gap(patient_id=patient_id, fhir_bundle=bundle)
        ranked.append(
            {
                "patient_id": patient_id,
                "priority_level": review["priority_level"],
                "gap_status": review.get("gap_status", "OPEN"),
                "clinical_reasoning": review["clinical_reasoning"],
                "equity_context": review["equity_context"],
                "delivery_date": review.get("delivery_date"),
                "follow_up_date": review.get("follow_up_date"),
            }
        )

    priority_order = {"URGENT": 0, "HIGH": 1, "MEDIUM": 2}
    ranked.sort(key=lambda x: (priority_order.get(x["priority_level"], 9), x["patient_id"]))

    return {
        "queue_size": len(ranked),
        "patients": ranked,
    }


# =============================================================================
# MCP FHIR RESOURCES
# =============================================================================


@mcp.resource("fhir://metadata")
async def metadata_resource() -> dict:
    return {
        "server_name": "FourthTrimesterCareAgent",
        "context_aware": True,
        "fhir_version": "R4",
        "supported_contexts": [
            "patient",
            "encounter",
            "practitioner",
        ],
    }


@mcp.resource("fhir://capability-statement")
async def capability_statement() -> dict:
    return {
        "resourceType": "CapabilityStatement",
        "status": "active",
        "kind": "instance",
        "fhirVersion": "4.0.1",
        "format": ["json"],
        "name": "FourthTrimesterCareAgent",
        "title": "Fourth Trimester Care Agent",
        "description": "FHIR-aware MCP server for postpartum maternal healthcare workflows.",
        "rest": [
            {
                "mode": "server",
                "resource": [
                    {"type": "Patient"},
                    {"type": "Observation"},
                    {"type": "Encounter"},
                    {"type": "Condition"},
                    {"type": "Procedure"},
                    {"type": "MedicationRequest"},
                    {"type": "CarePlan"},
                    {"type": "ServiceRequest"},
                    {"type": "Communication"},
                    {"type": "Task"},
                    {"type": "Appointment"},
                ]
            }
        ]
    }


@mcp.resource("fhir://Patient/{patient_id}")
async def patient_resource(patient_id: str) -> dict:
    return {
        "resourceType": "Patient",
        "id": patient_id,
        "active": True,
    }


@mcp.resource("fhir://Encounter/{encounter_id}")
async def encounter_resource(encounter_id: str) -> dict:
    return {
        "resourceType": "Encounter",
        "id": encounter_id,
        "status": "finished",
    }


# =============================================================================
# HEALTH + METADATA ENDPOINTS
# =============================================================================


@app.get("/")
async def root():
    return {
        "message": "Fourth Trimester Equity Agent is Live",
        "mcp_endpoint": "/mcp",
        "health_endpoint": "/healthz",
        "fhir_metadata": "/fhir-capabilities",
    }


@app.get("/healthz")
async def health():
    return {
        "status": "ok",
        "message": "Materna is listening",
    }


@app.get("/fhir-capabilities")
async def fhir_capabilities():
    return {
        "context_aware": True,
        "fhir_version": "R4",
        "transport": "streamable_http",
        "contexts_supported": [
            "patient",
            "encounter",
            "practitioner",
        ],
        "resources_supported": [
            "Patient",
            "Observation",
            "Encounter",
            "Condition",
            "Procedure",
            "MedicationRequest",
            "CarePlan",
            "ServiceRequest",
            "Communication",
            "Task",
            "Appointment",
        ],
        "tools": [
            "audit_postpartum_gap",
            "schedule_postpartum_visit",
            "send_outreach_message",
            "verify_follow_up_completion",
            "escalate_to_clinician",
            "generate_care_plan_summary",
            "analyze_sdoh_risks",
            "detect_postpartum_depression_screening_gap",
            "build_outreach_queue",
        ],
        "mcp_resources": [
            "fhir://metadata",
            "fhir://capability-statement",
            "fhir://Patient/{patient_id}",
            "fhir://Encounter/{encounter_id}",
        ],
    }


@app.get("/mcp/.well-known/fhir-context")
async def well_known_fhir_context():
    return {
        "server_name": "FourthTrimesterCareAgent",
        "context_aware": True,
        "fhir_version": "R4",
        "smart_on_fhir": False,
        "context_support": {
            "patient": True,
            "encounter": True,
            "practitioner": True,
        },
        "supported_resources": [
            "Patient",
            "Observation",
            "Encounter",
            "Condition",
            "Procedure",
            "MedicationRequest",
            "CarePlan",
            "ServiceRequest",
            "Communication",
            "Task",
            "Appointment",
        ],
        "supported_tools": [
            "audit_postpartum_gap",
            "schedule_postpartum_visit",
            "send_outreach_message",
            "verify_follow_up_completion",
            "escalate_to_clinician",
            "generate_care_plan_summary",
            "analyze_sdoh_risks",
            "detect_postpartum_depression_screening_gap",
            "build_outreach_queue",
        ],
    }


# =============================================================================
# MOUNT MCP SERVER
# =============================================================================

app.mount("/mcp", mcp_app)
