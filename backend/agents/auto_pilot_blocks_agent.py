from __future__ import annotations

from pathlib import Path
from typing import Any

from backend.agents.base import BaseAgent
from backend.llm.claude_client import ClaudeClient


def _load_prompt_template_text() -> str:
    path = (
        Path(__file__).resolve().parent.parent
        / "prompts"
        / "analysis_blocks_template_auto.json"
    )
    return path.read_text(encoding="utf-8")


_AUTO_TEMPLATE_TEXT = _load_prompt_template_text()

_AUTO_PILOT_PROMPT_BASE = """You are an Auto Pilot cloud architecture engine.

INPUTS
- user_input: business requirement text
- scope: signals from scope analysis (app_type/domain/stack/scale_hint/detected_signals)
- confirmed: the 3 confirmed answers (may be missing or messy text):
  - users
  - visibility: public|internal|both
  - uptime: 99%|99.9%|99.99%|99.999%
  - Additionally, confirmed MAY include deterministic derived fields:
    - concurrent_users_band (enum token)
    - public_facing (boolean)
    - sla_target (enum token)

TASK
- Fill the 7-block analysis JSON COMPLETELY in ONE call.
- Output ONLY valid JSON (no markdown, no comments, no explanation).
- No confidence object in Auto Pilot output.
- Every field must be set; default aggressively when missing.

TEMPLATE (CRITICAL)
- You are given a JSON TEMPLATE that contains the exact keys you must output.
- Your job is to REPLACE the placeholder values (strings like "<<...>>") with final values.
- Keep the SAME top-level keys and nested keys (no missing keys, no extra keys).
- Output MUST be valid JSON with correct types (booleans true/false, numbers where needed, null only where allowed).

COPY-ONLY ENUM RULE (CRITICAL)
- For EVERY enum field, choose the value by COPYING EXACTLY one token from the allowed lists below.
- Never invent new tokens.

ALLOWED ENUMS (copy exactly)
- app_type: web_application | api_service | data_pipeline | ml_model | ecommerce | real_time | devops_tool | other
- frontend_framework: react | vue | angular | nextjs | nuxt | svelte | other | null
- framework: django | laravel | nodejs | spring_boot | fastapi | rails | nextjs | other | null
- language: python | php | javascript | java | go | ruby | typescript | other | null
- database_engine: postgresql | mysql | mongodb | redis | elasticsearch | influxdb | clickhouse | sqlite | null
- domain: healthcare | fintech | ecommerce | media | devops | general | other
- pattern: single_vm | two_tier | three_tier | microservices | event_driven | data_pipeline | ml_pipeline
- message_queue_engine: rabbitmq | kafka | none
- cache_engine: redis | memcached | none
- network_mode: auto | manual
- concurrent_users_band: under_100 | 100_to_1k | 1k_to_10k | 10k_to_100k | 100k_plus
- data_storage_band: small_under_10gb | medium_10_to_500gb | large_500gb_plus
- traffic_pattern: steady | spiky_events | scheduled_batch | realtime_continuous
- downtime_impact: casual_acceptable | serious_issue | business_critical
- sla_target: best_effort | 99.9 | 99.99
- backup_policy: none | daily | hourly
- compliance_requirements items: HIPAA | PCI-DSS | GDPR | SOC2 | none
- sensitive_data_types items: payments | health_records | financial_data | pii | none
- tls_version: 1.2 | 1.3
- vm_size_preference: small_2vcpu_4gb | medium_4vcpu_8gb | large_8vcpu_16gb | xlarge_16vcpu_32gb
- environment: development | staging | production
- web_vm_type/app_vm_type/db_vm_type: small | medium | large | xlarge
- db_storage_gb: 20 | 50 | 100 | 200 | 500 | 1000

MINIMUM NORMALIZATION (only for the 3 confirmed answers)
- users -> concurrent_users_band:
  <100 under_100 | 100-999 100_to_1k | 1k-9,999 1k_to_10k | 10k-99,999 10k_to_100k | 100k+ 100k_plus
  If missing: default 1k_to_10k.
- uptime -> sla_target:
  99% -> best_effort | 99.9% -> 99.9 | 99.99% or 99.999% -> 99.99
  If missing: default 99.9.
- visibility -> public_facing:
  internal->false | public/both->true
  If missing: default true.
IMPORTANT OVERRIDE RULE
- If confirmed.concurrent_users_band is provided, you MUST use it for analysis_block_4_traffic_and_scale.concurrent_users_band.
- If confirmed.public_facing is provided, you MUST use it for analysis_block_6_access_and_security.public_facing.
- If confirmed.sla_target is provided, you MUST use it for analysis_block_5_availability.sla_target.

GUIDANCE
- Use your architecture knowledge, but default SIMPLE when user_input is vague.
- Prefer compliance_requirements=["none"] unless clearly triggered.
- Use standard network defaults (network_mode=auto and CIDRs exactly as skeleton).
- EVIDENCE-GATED COMPONENTS (IMPORTANT)
  - message_queue_required:
    - Set true ONLY if the requirement clearly implies asynchronous workloads (e.g., background jobs, email/SMS notifications, file processing, long-running tasks, event streaming, decoupled producers/consumers).
    - If it's only a small “send notification” feature and overall scale is small/medium, you may keep it false (simpler) unless explicitly requested.
  - message_queue_engine:
    - If message_queue_required=false then message_queue_engine MUST be "none".
    - If message_queue_required=true choose kafka ONLY for high-throughput/event-streaming needs; otherwise prefer rabbitmq for typical task queues.
  - cache_required:
    - Set true ONLY if the requirement implies cacheable hot reads (search/browse/catalog), high read traffic, or performance/latency sensitivity.
    - Otherwise set cache_required=false and cache_engine="none".
  - cache_engine:
    - If cache_required=false then cache_engine MUST be "none".
    - If cache_required=true prefer redis unless there is a specific reason for memcached.
  - audit_log_required (CRITICAL CONSISTENCY RULE):
    - Set true IF AND ONLY IF compliance_requirements contains at least one of: HIPAA, PCI-DSS, SOC2.
    - Set false for all other cases, including GDPR-only, or compliance_requirements=["none"].
- RECOMMEND STACK VALUES (IMPORTANT)
  - You MUST fill these fields based on the business requirement (do not leave them null when the app clearly needs them):
    - frontend_framework:
      - If app_type is one of: web_application | ecommerce | real_time | devops_tool
        then frontend_framework MUST be one of the allowed non-null tokens.
      - Only set frontend_framework=null when app_type is api_service | data_pipeline | ml_model (no UI).
    - framework (backend):
      - For app_type web_application | ecommerce | real_time | devops_tool | api_service
        then framework MUST be one of the allowed non-null tokens.
      - Only set framework=null when app_type is data_pipeline or ml_model AND the requirement clearly implies no web/API backend.
    - language:
      - If framework is not null, language MUST NOT be null and MUST be consistent with the chosen framework.
  - You still choose WHICH framework to use from the enum list using your own knowledge; just do not leave these blank.

DECIDE VALUES FOR ALL BLOCKS (IMPORTANT)
- Do NOT leave any required field at a “generic” value just because it’s convenient.
- For optional-or-nullable fields, only use null when the business clearly implies the component does not exist.
  Examples:
  - If app_type implies a UI, frontend_framework must not be null.
  - If database_required=true then database_engine must not be null.
  - If handles_sensitive_data=true then sensitive_data_types must not be ["none"].
- For fields that are unclear, pick the most reasonable enum using your own knowledge (and keep it simple).
- SCALE COHERENCE (IMPORTANT)
  - resolved_scale must be coherent with concurrent_users_band and scope.scale_hint:
    - under_100 -> small
    - 100_to_1k -> small or medium (choose medium if public-facing, ecommerce, real_time, or expected_growth=true)
    - 1k_to_10k -> medium
    - 10k_to_100k -> large
    - 100k_plus -> large

OUTPUT SHAPE (must match exactly)
- Top-level JSON must contain exactly these 7 keys:
  - analysis_block_1_application_identity
  - analysis_block_2_architecture_pattern
  - analysis_block_3_network_design
  - analysis_block_4_traffic_and_scale
  - analysis_block_5_availability
  - analysis_block_6_access_and_security
  - analysis_block_7_resource_sizing_and_budget
- Each block must contain ALL required keys from the schema (no missing keys, no extra keys).
- Network block must use the standard CIDR strings shown below (do not change them):
  vpc_cidr=10.0.0.0/16, public_subnet_cidr=10.0.1.0/24, private_subnet_cidr=10.0.2.0/24, db_subnet_cidr=10.0.3.0/24

JSON TEMPLATE (fill this exact shape; replace placeholder values)
{template_json}

DO NOT include a pre-filled example JSON in your output. You must generate the full JSON with your chosen values.
"""

AUTO_PILOT_PROMPT = (
    _AUTO_PILOT_PROMPT_BASE
    + "\n\nJSON TEMPLATE (fill this exact shape; replace placeholder values)\n"
    + _AUTO_TEMPLATE_TEXT
    + "\n"
)


class AutoPilotBlocksAgent(BaseAgent):
    def __init__(self, client: ClaudeClient) -> None:
        super().__init__(client)

    async def run(
        self,
        *,
        user_input: str,
        scope: dict[str, Any],
        confirmed: dict[str, Any],
    ) -> dict[str, Any]:
        ctx = {
            "user_input": user_input,
            "scope": scope,
            "confirmed": confirmed,
        }
        return await self.client.generate_json(
            system_prompt=AUTO_PILOT_PROMPT,
            user_message=f"Fill the 7 analysis blocks from this context:\n\n{ctx}",
            max_tokens=2000,
        )

