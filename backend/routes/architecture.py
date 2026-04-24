from fastapi import APIRouter, HTTPException, Depends
from functools import lru_cache

from backend.models.schemas import (
    ScopeRequest, ScopeResponse,
    ModeRequest, ModeResponse,
    AutoPilotInitRequest, AutoPilotInitResponse,
    AutoPilotCompleteRequest,
    GuidedBlockRequest, GuidedBlockResponse,
    GuidedCompleteRequest,
    SmartGuidedQuestionsRequest, SmartGuidedQuestionsResponse,
    SmartGuidedCompleteRequest,
    ConversationalMessageRequest, ConversationalMessageResponse,
    GuidedAnalysisRequest, GuidedAnalysisBlocks,
    GuidedLoopStartRequest, GuidedLoopStartResponse,
    GuidedLoopAnswerRequest, GuidedLoopAnswerResponse,
    GuidedLoopSessionState, GuidedLoopQuestion, AnalysisConfidence,
    SolutionOutput, ArchitectureSummary, ReasoningLog, CytoscapeElements,
    BusinessRequirements,
)
from backend.llm.claude_client import ClaudeClient
from backend.agents.scope_agent import ScopeAgent
from backend.agents.mode_selection_agent import ModeSelectionAgent
from backend.agents.auto_pilot_blocks_agent import AutoPilotBlocksAgent
from backend.agents.business_requirement_agent import BusinessRequirementAgent
from backend.agents.guided_agent import GuidedAgent
from backend.agents.smart_guided_agent import SmartGuidedAgent
from backend.agents.conversational_guided_agent import ConversationalGuidedAgent
from backend.agents.guided_analysis_agent import GuidedAnalysisAgent
from backend.agents.question_loop_agent import QuestionLoopAgent
from backend.agents.template_agent import TemplateAgent
from backend.agents.reasoning_agent import ReasoningAgent
from pydantic import ValidationError
from backend.utils.guided_loop_store import GuidedLoopStore
from backend.utils.analysis_consistency import check_analysis_blocks_consistency

router = APIRouter(prefix="/api", tags=["architecture"])

_guided_loop_store = GuidedLoopStore(ttl_seconds=3600)


@lru_cache(maxsize=1)
def _client() -> ClaudeClient:
    try:
        return ClaudeClient()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _get_client():
    return _client()


# ── Health ──────────────────────────────────────────────────────────────────
@router.get("/health")
async def health():
    from backend.config import CLAUDE_API_KEY
    return {
        "status": "ok",
        "api_key_configured": bool(CLAUDE_API_KEY),
    }


# ── Scope ───────────────────────────────────────────────────────────────────
@router.post("/scope", response_model=ScopeResponse)
async def analyze_scope(req: ScopeRequest):
    try:
        client = _get_client()
        agent = ScopeAgent(client)
        result = await agent.run(user_input=req.user_input)
        return ScopeResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Scope analysis failed: {exc}") from exc


# ── Mode ────────────────────────────────────────────────────────────────────
@router.post("/mode", response_model=ModeResponse)
async def select_mode(req: ModeRequest):
    try:
        client = _get_client()
        agent = ModeSelectionAgent(client)
        result = await agent.run(
            user_input=req.user_input,
            scope=req.scope.model_dump(),
        )
        return ModeResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Mode selection failed: {exc}") from exc


# ── Auto Mode ───────────────────────────────────────────────────────────────
@router.post("/auto/init", response_model=AutoPilotInitResponse)
async def auto_init(req: AutoPilotInitRequest):
    try:
        # Auto-pilot init is deterministic: show confirmed detections from scope + ask 3 fixed questions.
        scope = req.scope.model_dump()
        confirmed = {
            "app_type": scope.get("app_type", "web"),
            "stack": scope.get("stack", []),
            "architecture_pattern": "three-tier",
            "session_store": "stateless",
            "deployment_model": "cloud-native",
        }
        missing_fields = [
            {
                "id": "users",
                "label": "Expected Users",
                "question": "How many concurrent users do you expect at peak?",
                "placeholder": "e.g., 500, 10000, 1M",
                "type": "text",
                "options": [],
            },
            {
                "id": "visibility",
                "label": "Visibility",
                "question": "Will this application be accessible from the internet?",
                "placeholder": "",
                "type": "select",
                "options": ["public", "internal", "both"],
            },
            {
                "id": "uptime",
                "label": "Uptime Requirement",
                "question": "What uptime level does your business need?",
                "placeholder": "",
                "type": "select",
                "options": ["99%", "99.9%", "99.99%", "99.999%"],
            },
        ]
        return AutoPilotInitResponse(
            confirmed_detections=confirmed,
            missing_fields=missing_fields,
            architecture_hint="Answer 3 questions and Auto Pilot will generate the full architecture.",
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auto-pilot init failed: {exc}") from exc


@router.post("/auto/complete", response_model=SolutionOutput)
async def auto_complete(req: AutoPilotCompleteRequest):
    try:
        client = _get_client()

        # Deterministic derivations from the 3 quick inputs.
        users_num = _parse_users(str(req.quick_inputs.users))
        if users_num < 100:
            users_band = "under_100"
        elif users_num < 1000:
            users_band = "100_to_1k"
        elif users_num < 10_000:
            users_band = "1k_to_10k"
        elif users_num < 100_000:
            users_band = "10k_to_100k"
        else:
            users_band = "100k_plus"

        uptime_raw = str(req.quick_inputs.uptime or "").strip()
        if uptime_raw == "99%":
            sla_target = "best_effort"
        elif uptime_raw == "99.9%":
            sla_target = "99.9"
        else:
            # Treat 99.99% and 99.999% (and any unknown high-uptime) as 99.99.
            sla_target = "99.99"

        visibility_raw = str(req.quick_inputs.visibility or "").strip().lower()
        if visibility_raw == "internal":
            public_facing = False
        else:
            # public | both | unknown -> true
            public_facing = True

        confirmed = {
            "users": req.quick_inputs.users,
            "visibility": req.quick_inputs.visibility,
            "uptime": req.quick_inputs.uptime,
            # Pass deterministic, schema-level hints so the LLM cannot "mis-map" them.
            "concurrent_users_band": users_band,
            "sla_target": sla_target,
            "public_facing": public_facing,
            **(req.auto_pilot_init.confirmed_detections.model_dump() if req.auto_pilot_init else {}),
        }

        blocks_agent = AutoPilotBlocksAgent(client)
        last_err: Exception | None = None
        raw_blocks: dict = {}
        for _attempt in range(3):
            raw_blocks = await blocks_agent.run(
                user_input=req.user_input,
                scope=req.scope.model_dump(),
                confirmed=confirmed,
            )
            # Hard-apply deterministic mappings from the 3 quick inputs.
            # This guarantees the final 7-block JSON cannot contradict user-confirmed answers.
            try:
                raw_blocks.setdefault("analysis_block_4_traffic_and_scale", {})["concurrent_users_band"] = users_band
                raw_blocks.setdefault("analysis_block_5_availability", {})["sla_target"] = sla_target
                raw_blocks.setdefault("analysis_block_6_access_and_security", {})["public_facing"] = public_facing
            except Exception:
                # If the model returned a malformed shape, let validation handle it.
                pass
            try:
                analysis_blocks = GuidedAnalysisBlocks(**raw_blocks)
                violations = check_analysis_blocks_consistency(analysis_blocks)
                if not violations:
                    break
                last_err = Exception("Consistency violations: " + "; ".join(vv.message for vv in violations))
                confirmed = {
                    **confirmed,
                    "_instructions": (
                        "Your previous JSON violated consistency rules. Output corrected JSON ONLY. "
                        "Change only the minimum fields needed to satisfy the listed violations; keep everything else identical."
                    ),
                    "_previous_json": raw_blocks,
                    "_consistency_violations": [vv.message for vv in violations],
                }
            except ValidationError as exc:
                last_err = exc
                confirmed = {
                    **confirmed,
                    "_validation_error": str(exc),
                    "_instructions": (
                        "Your previous JSON failed schema validation. Output corrected JSON ONLY. "
                        "Change only the minimum fields needed to satisfy the error; keep everything else identical."
                    ),
                    "_previous_json": raw_blocks,
                    "_consistency_violations": [vv.message for vv in (violations if "violations" in locals() else [])],
                }
            except Exception as exc:
                last_err = exc
                confirmed = {
                    **confirmed,
                    "_instructions": (
                        "Your previous JSON violated consistency rules. Output corrected JSON ONLY. "
                        "Change only the minimum fields needed to satisfy the listed violations; keep everything else identical."
                    ),
                    "_previous_json": raw_blocks,
                    "_consistency_violations": [vv.message for vv in (violations if "violations" in locals() else [])],
                }
        else:
            raise HTTPException(status_code=500, detail=f"Auto-pilot blocks validation failed: {last_err}")

        requirements = _analysis_blocks_to_requirements(analysis_blocks, req.scope.model_dump())

        tmpl_agent = TemplateAgent(client)
        template_result = await tmpl_agent.run(requirements=requirements)

        rsn_agent = ReasoningAgent(client)
        reasoning_result = await rsn_agent.run(
            template_result=template_result,
            requirements=requirements,
        )

        return _build_solution(template_result, reasoning_result, requirements, analysis_blocks=analysis_blocks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Auto-pilot complete failed: {exc}") from exc


# ── Guided Mode ─────────────────────────────────────────────────────────────
@router.post("/guided/questions/{block}", response_model=GuidedBlockResponse)
async def guided_questions(block: int, req: GuidedBlockRequest):
    if block < 1 or block > 7:
        raise HTTPException(status_code=400, detail="Block must be between 1 and 7")
    try:
        client = _get_client()
        agent = GuidedAgent(client)
        result = await agent.get_block_questions(
            block=block,
            user_input=req.user_input,
            scope=req.scope.model_dump(),
            previous_answers=req.previous_answers,
        )
        return GuidedBlockResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Guided block {block} failed: {exc}") from exc


@router.post("/guided/complete", response_model=SolutionOutput)
async def guided_complete(req: GuidedCompleteRequest):
    try:
        client = _get_client()

        flat_answers = {}
        for block_answers in req.answers.values():
            if isinstance(block_answers, dict):
                flat_answers.update(block_answers)

        analysis_blocks = await _validated_guided_analysis_blocks(
            client,
            user_input=req.user_input,
            scope=req.scope.model_dump(),
            answers=flat_answers,
        )
        requirements = _analysis_blocks_to_requirements(analysis_blocks, req.scope.model_dump())

        tmpl_agent = TemplateAgent(client)
        template_result = await tmpl_agent.run(requirements=requirements)

        rsn_agent = ReasoningAgent(client)
        reasoning_result = await rsn_agent.run(
            template_result=template_result,
            requirements=requirements,
        )

        return _build_solution(template_result, reasoning_result, requirements, analysis_blocks=analysis_blocks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Guided complete failed: {exc}") from exc


@router.post("/guided/smart/questions", response_model=SmartGuidedQuestionsResponse)
async def smart_guided_questions(req: SmartGuidedQuestionsRequest):
    """
    Smart Guided Mode — Step 1.
    Pre-fills what it can from scope, then generates the minimum questions
    (always 3-7) the user must answer to complete the 7 analysis blocks.
    """
    try:
        client = _get_client()
        agent = SmartGuidedAgent(client)
        result = await agent.run(
            user_input=req.user_input,
            scope=req.scope.model_dump(),
        )
        return SmartGuidedQuestionsResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Smart guided questions failed: {exc}") from exc


@router.post("/guided/smart/complete", response_model=SolutionOutput)
async def smart_guided_complete(req: SmartGuidedCompleteRequest):
    """
    Smart Guided Mode — Step 2.
    Takes user answers from the smart questions, fills all 7 analysis blocks,
    and generates the architecture solution.
    """
    try:
        client = _get_client()
        analysis_blocks = await _validated_guided_analysis_blocks(
            client,
            user_input=req.user_input,
            scope=req.scope.model_dump(),
            answers=req.answers,
        )
        requirements = _analysis_blocks_to_requirements(analysis_blocks, req.scope.model_dump())

        tmpl_agent = TemplateAgent(client)
        template_result = await tmpl_agent.run(requirements=requirements)

        rsn_agent = ReasoningAgent(client)
        reasoning_result = await rsn_agent.run(
            template_result=template_result,
            requirements=requirements,
        )

        return _build_solution(template_result, reasoning_result, requirements, analysis_blocks=analysis_blocks)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Smart guided complete failed: {exc}") from exc


@router.post("/guided/conversation/message", response_model=ConversationalMessageResponse)
async def conversational_message(req: ConversationalMessageRequest):
    """
    Conversational Guided Mode — send one message, get one question back.
    When the LLM has collected enough information, returns is_complete=true
    and collected_answers. Frontend then calls /guided/smart/complete.
    """
    try:
        client = _get_client()
        agent = ConversationalGuidedAgent(client)
        result = await agent.run(
            user_input=req.user_input,
            scope=req.scope.model_dump(),
            conversation=[t.model_dump() for t in req.conversation],
            user_message=req.message,
        )
        return ConversationalMessageResponse(**result)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Conversational message failed: {exc}") from exc


# ── Guided Loop Mode (confidence-driven, multi-turn) ─────────────────────────
@router.post("/guided/loop/start", response_model=GuidedLoopStartResponse)
async def guided_loop_start(req: GuidedLoopStartRequest):
    """
    Start a new guided-loop session:
    - Run analysis (7 blocks + confidence) with no answers
    - Run question agent to pick first question (or stop)
    - Persist session in store
    """
    try:
        client = _get_client()
        session_id = _guided_loop_store.new_session_id()

        analysis_agent = GuidedAnalysisAgent(client)
        analysis_raw: dict = {}
        last_err: Exception | None = None
        answers: dict = {}
        for _attempt in range(2):
            analysis_raw = await analysis_agent.run(
                user_input=req.user_input,
                scope=req.scope.model_dump(),
                answers=answers,
            )
            confidence = analysis_raw.get("confidence", {})
            blocks_raw = {k: v for k, v in analysis_raw.items() if k != "confidence"}
            try:
                blocks_model = GuidedAnalysisBlocks(**blocks_raw)
                violations = check_analysis_blocks_consistency(blocks_model)
                if violations:
                    last_err = Exception("Consistency violations: " + "; ".join(vv.message for vv in violations))
                    raise last_err
                break
            except ValidationError as exc:
                last_err = exc
                answers = {
                    **(answers or {}),
                    "_validation_error": str(exc),
                    "_instructions": "Your previous JSON failed schema validation. Output corrected JSON ONLY, using allowed enum values and correct types.",
                    "_previous_json": blocks_raw,
                    "_consistency_violations": [vv.message for vv in (violations if "violations" in locals() else [])],
                }
            except Exception as exc:
                last_err = exc
                answers = {
                    **(answers or {}),
                    "_instructions": "Your previous JSON violated consistency rules. Output corrected JSON ONLY.",
                    "_previous_json": blocks_raw,
                    "_consistency_violations": [vv.message for vv in (violations if "violations" in locals() else [])],
                }
        else:
            raise HTTPException(status_code=500, detail=f"Guided loop start validation failed: {last_err}")

        q_agent = QuestionLoopAgent(client)
        q_raw = await q_agent.run(
            user_input=req.user_input,
            answers={},
            analysis_blocks=blocks_raw,
            confidence=confidence,
            question_count=0,
            max_questions=req.max_questions,
        )

        state = GuidedLoopSessionState(
            session_id=session_id,
            user_input=req.user_input,
            scope=req.scope,
            answers={},
            question_count=0,
            max_questions=req.max_questions,
            analysis_blocks=blocks_model,
            confidence=AnalysisConfidence(**confidence) if confidence else None,
            status="questioning",
        )

        if q_raw.get("action") == "stop":
            state.status = "complete"
            _guided_loop_store.set(session_id, state.model_dump())
            return GuidedLoopStartResponse(
                session_id=session_id,
                status="complete",
                analysis_blocks=state.analysis_blocks,
                confidence=state.confidence,
            )

        question = q_raw.get("question") or {}
        _guided_loop_store.set(session_id, state.model_dump())
        return GuidedLoopStartResponse(
            session_id=session_id,
            status="questioning",
            question=GuidedLoopQuestion(**question),
            question_number=1,
            analysis_blocks=state.analysis_blocks,
            confidence=state.confidence,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Guided loop start failed: {exc}") from exc


@router.post("/guided/loop/answer", response_model=GuidedLoopAnswerResponse)
async def guided_loop_answer(req: GuidedLoopAnswerRequest):
    """
    Submit one answer:
    - Update session answers
    - Run analysis (7 blocks + confidence)
    - Run question agent (stop vs ask next)
    - Persist session
    """
    try:
        raw_state = _guided_loop_store.get(req.session_id)
        if not raw_state:
            raise HTTPException(status_code=404, detail="Session not found")

        state = GuidedLoopSessionState(**raw_state)
        if state.status == "complete":
            return GuidedLoopAnswerResponse(
                session_id=state.session_id,
                status="complete",
                analysis_blocks=state.analysis_blocks,
                confidence=state.confidence,
            )

        # update
        state.answers[req.field] = req.value
        state.question_count += 1

        client = _get_client()
        # Provide previous low-confidence fields to encourage monotonic improvement per prompt rule.
        prev_low = []
        if state.confidence and state.confidence.low_confidence_fields:
            prev_low = list(state.confidence.low_confidence_fields)

        analysis_agent = GuidedAnalysisAgent(client)
        analysis_raw: dict = {}
        last_err: Exception | None = None
        answers = {**state.answers, "_previous_low_confidence_fields": prev_low}
        for _attempt in range(2):
            analysis_raw = await analysis_agent.run(
                user_input=state.user_input,
                scope=state.scope.model_dump(),
                answers=answers,
            )
            confidence = analysis_raw.get("confidence", {})
            blocks_raw = {k: v for k, v in analysis_raw.items() if k != "confidence"}
            try:
                state.analysis_blocks = GuidedAnalysisBlocks(**blocks_raw)
                violations = check_analysis_blocks_consistency(state.analysis_blocks)
                if violations:
                    last_err = Exception("Consistency violations: " + "; ".join(vv.message for vv in violations))
                    raise last_err
                break
            except ValidationError as exc:
                last_err = exc
                answers = {
                    **answers,
                    "_validation_error": str(exc),
                    "_instructions": "Your previous JSON failed schema validation. Output corrected JSON ONLY, using allowed enum values and correct types.",
                    "_previous_json": blocks_raw,
                    "_consistency_violations": [vv.message for vv in (violations if "violations" in locals() else [])],
                }
            except Exception as exc:
                last_err = exc
                answers = {
                    **answers,
                    "_instructions": "Your previous JSON violated consistency rules. Output corrected JSON ONLY.",
                    "_previous_json": blocks_raw,
                    "_consistency_violations": [vv.message for vv in (violations if "violations" in locals() else [])],
                }
        else:
            raise HTTPException(status_code=500, detail=f"Guided loop answer validation failed: {last_err}")

        state.confidence = AnalysisConfidence(**confidence) if confidence else None

        q_agent = QuestionLoopAgent(client)
        q_raw = await q_agent.run(
            user_input=state.user_input,
            answers=state.answers,
            analysis_blocks=blocks_raw,
            confidence=confidence,
            question_count=state.question_count,
            max_questions=state.max_questions,
        )

        if q_raw.get("action") == "stop":
            state.status = "complete"
            _guided_loop_store.set(state.session_id, state.model_dump())
            return GuidedLoopAnswerResponse(
                session_id=state.session_id,
                status="complete",
                analysis_blocks=state.analysis_blocks,
                confidence=state.confidence,
            )

        question = q_raw.get("question") or {}
        _guided_loop_store.set(state.session_id, state.model_dump())
        return GuidedLoopAnswerResponse(
            session_id=state.session_id,
            status="questioning",
            question=GuidedLoopQuestion(**question),
            question_number=state.question_count + 1,
            analysis_blocks=state.analysis_blocks,
            confidence=state.confidence,
        )
    except ValidationError as exc:
        raise HTTPException(status_code=500, detail=f"Guided loop answer validation failed: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Guided loop answer failed: {exc}") from exc


@router.get("/guided/loop/session/{session_id}", response_model=GuidedLoopSessionState)
async def guided_loop_session(session_id: str):
    raw_state = _guided_loop_store.get(session_id)
    if not raw_state:
        raise HTTPException(status_code=404, detail="Session not found")
    return GuidedLoopSessionState(**raw_state)


@router.post("/guided/analysis", response_model=GuidedAnalysisBlocks)
async def guided_analysis(req: GuidedAnalysisRequest):
    """
    Generate the 7-block analysis JSON (filled enum values) from user input,
    scope, and any previously answered guided questions.
    """
    try:
        client = _get_client()
        return await _validated_guided_analysis_blocks(
            client,
            user_input=req.user_input,
            scope=req.scope.model_dump(),
            answers=req.answers,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Guided analysis failed: {exc}") from exc


# ── Helpers ──────────────────────────────────────────────────────────────────
def _build_solution(
    template_result: dict,
    reasoning_result: dict,
    requirements: dict,
    analysis_blocks: GuidedAnalysisBlocks | None = None,
) -> SolutionOutput:
    summary_data = reasoning_result.get("summary", {})
    reasoning_data = reasoning_result.get("reasoning", {})
    cy_raw = template_result.get("cytoscape_elements", {"nodes": [], "edges": []})

    return SolutionOutput(
        template_id=template_result["template_id"],
        template_name=template_result["template_name"],
        summary=ArchitectureSummary(
            architecture_type=summary_data.get("architecture_type", template_result["template_name"]),
            components=summary_data.get("components", []),
            estimated_monthly_cost=summary_data.get("estimated_monthly_cost", "N/A"),
            deployment_complexity=summary_data.get("deployment_complexity", "Medium"),
            key_highlights=summary_data.get("key_highlights", []),
        ),
        reasoning=ReasoningLog(
            template_selection=reasoning_data.get("template_selection", ""),
            component_choices=reasoning_data.get("component_choices", []),
            trade_offs=reasoning_data.get("trade_offs", []),
            alternatives_considered=reasoning_data.get("alternatives_considered", []),
        ),
        configuration=template_result.get("configuration", {}),
        cytoscape_elements=CytoscapeElements(
            nodes=cy_raw.get("nodes", []),
            edges=cy_raw.get("edges", []),
        ),
        requirements=BusinessRequirements(**requirements),
        analysis_blocks=analysis_blocks,
    )


def _analysis_blocks_to_requirements(blocks: GuidedAnalysisBlocks, scope: dict) -> dict:
    b1 = blocks.analysis_block_1_application_identity
    b2 = blocks.analysis_block_2_architecture_pattern
    b4 = blocks.analysis_block_4_traffic_and_scale
    b5 = blocks.analysis_block_5_availability
    b6 = blocks.analysis_block_6_access_and_security

    users_map = {
        "under_100": 50,
        "100_to_1k": 500,
        "1k_to_10k": 5000,
        "10k_to_100k": 50000,
        "100k_plus": 200000,
    }
    users = users_map.get(b4.concurrent_users_band, 1000)

    uptime_map = {"best_effort": "99%", "99.9": "99.9%", "99.99": "99.99%"}
    uptime = uptime_map.get(b5.sla_target, "99.9%")

    classification = "internal"
    if b6.handles_sensitive_data or any(x in (b6.sensitive_data_types or []) for x in ("payments", "health_records", "financial_data", "pii")):
        classification = "confidential"

    tier = "growing"
    if b7 := blocks.analysis_block_7_resource_sizing_and_budget:
        if (b7.monthly_budget_ceiling_usd or 0) >= 10000:
            tier = "enterprise"
        elif (b7.monthly_budget_ceiling_usd or 0) > 0 and (b7.monthly_budget_ceiling_usd or 0) < 500:
            tier = "startup"

    return {
        "app_type": b1.app_type,
        "scale": {
            "concurrent_users": int(users),
            "requests_per_second": max(1, int(users) // 10),
            "storage_tb": 0.1,
            "growth_rate": "growing" if b4.expected_growth else "steady",
        },
        "availability": {
            "uptime_requirement": uptime,
            "rto_minutes": 15 if b5.sla_target == "99.99" else 60,
            "rpo_minutes": 30,
            "multi_az": bool(b5.multi_az),
        },
        "network": {
            "internet_facing": bool(b6.public_facing),
            "cdn_required": bool(b6.public_facing and users > 1000),
            "multi_region": bool(b5.disaster_recovery_required),
        },
        "security": {
            "authentication": True,
            "data_classification": classification,
            "compliance": [c for c in (b6.compliance_requirements or []) if c != "none"],
        },
        "budget": {
            "tier": tier,
            "monthly_estimate_usd": int(blocks.analysis_block_7_resource_sizing_and_budget.monthly_budget_ceiling_usd or 0),
            "cost_optimization": "balanced",
        },
        "stack": scope.get("stack", []),
        "derived_requirements": [
            f"Derived from 7-block analysis (pattern={b2.pattern}, scale={b4.resolved_scale}, sla={b5.sla_target})"
        ],
    }


async def _validated_guided_analysis_blocks(
    client: ClaudeClient,
    *,
    user_input: str,
    scope: dict,
    answers: dict,
    max_attempts: int = 3,
) -> GuidedAnalysisBlocks:
    """
    Ask the GuidedAnalysisAgent for 7-blocks (+ confidence) and validate blocks via Pydantic.
    If validation fails, retry once with the validation error so the LLM can correct enums/types.
    """
    agent = GuidedAnalysisAgent(client)
    last_err: Exception | None = None
    raw: dict = {}

    for _attempt in range(max_attempts):
        raw = await agent.run(user_input=user_input, scope=scope, answers=answers)
        blocks_raw = {k: v for k, v in raw.items() if k != "confidence"}
        try:
            blocks = GuidedAnalysisBlocks(**blocks_raw)
            violations = check_analysis_blocks_consistency(blocks)
            if violations:
                last_err = Exception("Consistency violations: " + "; ".join(vv.message for vv in violations))
                raise last_err
            return blocks
        except ValidationError as exc:
            last_err = exc
            answers = {
                **(answers or {}),
                "_validation_error": str(exc),
                "_instructions": "Your previous JSON failed schema validation. Output corrected JSON ONLY, using allowed enum values and correct types.",
                "_previous_json": blocks_raw,
                "_consistency_violations": [vv.message for vv in (violations if "violations" in locals() else [])],
            }
        except Exception as exc:
            last_err = exc
            answers = {
                **(answers or {}),
                "_instructions": "Your previous JSON violated consistency rules. Output corrected JSON ONLY.",
                "_previous_json": blocks_raw,
                "_consistency_violations": [vv.message for vv in (violations if "violations" in locals() else [])],
            }

    raise HTTPException(status_code=500, detail=f"Guided analysis block validation failed: {last_err}")


def _guided_answers_to_requirements(answers: dict, scope: dict) -> dict:
    users_str = str(answers.get("users_count", answers.get("concurrent_users", "1000")))
    users_num = _parse_users(users_str)

    uptime_map = {
        "hours": "99%",
        "30 minutes": "99.9%",
        "minutes": "99.99%",
        "cannot tolerate": "99.999%",
    }
    downtime_raw = str(answers.get("downtime_tolerance", "")).lower()
    uptime = next((v for k, v in uptime_map.items() if k in downtime_raw), "99.9%")

    internet = str(answers.get("internet_access", "yes")).lower()
    internet_facing = "internal" not in internet

    sensitive = answers.get("sensitive_data", [])
    if isinstance(sensitive, str):
        sensitive = [sensitive]
    has_pii = any("personal" in s.lower() or "pii" in s.lower() for s in sensitive)
    has_payment = any("payment" in s.lower() or "pci" in s.lower() for s in sensitive)
    has_health = any("health" in s.lower() or "hipaa" in s.lower() for s in sensitive)

    compliance = []
    if has_payment:
        compliance.append("PCI-DSS")
    if has_health:
        compliance.append("HIPAA")

    budget_map = {
        "<$100": ("startup", 50),
        "$100-$500": ("startup", 300),
        "$500-$2000": ("growing", 1250),
        "$2000-$10000": ("growing", 6000),
        ">$10000": ("enterprise", 15000),
    }
    budget_raw = answers.get("monthly_budget", "$500-$2000")
    tier, estimate = budget_map.get(budget_raw, ("growing", 1250))

    return {
        "app_type": scope.get("app_type", "web"),
        "scale": {
            "concurrent_users": users_num,
            "requests_per_second": max(1, users_num // 10),
            "storage_tb": 0.1,
            "growth_rate": "growing",
        },
        "availability": {
            "uptime_requirement": uptime,
            "rto_minutes": 60 if "99%" == uptime else 15,
            "rpo_minutes": 30,
            "multi_az": users_num > 500 or uptime in ("99.99%", "99.999%"),
        },
        "network": {
            "internet_facing": internet_facing,
            "cdn_required": internet_facing and users_num > 1000,
            "multi_region": False,
        },
        "security": {
            "authentication": True,
            "data_classification": "confidential" if (has_pii or has_payment or has_health) else "internal",
            "compliance": compliance,
        },
        "budget": {
            "tier": tier,
            "monthly_estimate_usd": estimate,
            "cost_optimization": "balanced",
        },
        "stack": scope.get("stack", []),
        "derived_requirements": [f"Derived from guided interview with {len(answers)} answers"],
    }


def _parse_users(raw: str) -> int:
    raw = raw.lower().replace(",", "").strip()
    if "m" in raw:
        try:
            return int(float(raw.replace("m", "")) * 1_000_000)
        except ValueError:
            pass
    if "k" in raw:
        try:
            return int(float(raw.replace("k", "")) * 1_000)
        except ValueError:
            pass
    import re
    nums = re.findall(r"\d+", raw)
    return int(nums[0]) if nums else 1000


