from __future__ import annotations

from dataclasses import dataclass


from backend.models.schemas import GuidedAnalysisBlocks


@dataclass(frozen=True)
class ConsistencyViolation:
    rule: str
    fields: list[str]
    message: str


def check_analysis_blocks_consistency(blocks: GuidedAnalysisBlocks) -> list[ConsistencyViolation]:
    """
    Generic cross-field invariants that should always hold.
    These are not "architecture choices"; they prevent contradictory outputs.
    """

    v: list[ConsistencyViolation] = []

    b2 = blocks.analysis_block_2_architecture_pattern
    b3 = blocks.analysis_block_3_network_design
    b4 = blocks.analysis_block_4_traffic_and_scale
    b5 = blocks.analysis_block_5_availability
    b6 = blocks.analysis_block_6_access_and_security
    b7 = blocks.analysis_block_7_resource_sizing_and_budget

    # NAT gateway: true except for single_vm pattern.
    if b2.pattern != "single_vm" and b3.nat_gateway is False:
        v.append(
            ConsistencyViolation(
                rule="nat_gateway_non_single_vm",
                fields=["analysis_block_2_architecture_pattern.pattern", "analysis_block_3_network_design.nat_gateway"],
                message="nat_gateway must be true for all patterns except single_vm.",
            )
        )

    # Message queue flags.
    if b2.message_queue_required is False and b2.message_queue_engine != "none":
        v.append(
            ConsistencyViolation(
                rule="queue_engine_none_when_disabled",
                fields=[
                    "analysis_block_2_architecture_pattern.message_queue_required",
                    "analysis_block_2_architecture_pattern.message_queue_engine",
                ],
                message='message_queue_required=false implies message_queue_engine="none".',
            )
        )
    if b2.message_queue_required is True and b2.message_queue_engine == "none":
        v.append(
            ConsistencyViolation(
                rule="queue_engine_set_when_enabled",
                fields=[
                    "analysis_block_2_architecture_pattern.message_queue_required",
                    "analysis_block_2_architecture_pattern.message_queue_engine",
                ],
                message='message_queue_required=true implies message_queue_engine is kafka or rabbitmq.',
            )
        )

    # Cache flags.
    if b2.cache_required is False and b2.cache_engine != "none":
        v.append(
            ConsistencyViolation(
                rule="cache_engine_none_when_disabled",
                fields=["analysis_block_2_architecture_pattern.cache_required", "analysis_block_2_architecture_pattern.cache_engine"],
                message='cache_required=false implies cache_engine="none".',
            )
        )
    if b2.cache_required is True and b2.cache_engine == "none":
        v.append(
            ConsistencyViolation(
                rule="cache_engine_set_when_enabled",
                fields=["analysis_block_2_architecture_pattern.cache_required", "analysis_block_2_architecture_pattern.cache_engine"],
                message="cache_required=true implies cache_engine is redis or memcached.",
            )
        )

    # Audit logging: only when compliance requires it.
    compliance = set(b6.compliance_requirements or [])
    audit_should_be_true = bool(compliance.intersection({"HIPAA", "PCI-DSS", "SOC2"}))
    if b6.audit_log_required != audit_should_be_true:
        v.append(
            ConsistencyViolation(
                rule="audit_log_from_compliance",
                fields=["analysis_block_6_access_and_security.compliance_requirements", "analysis_block_6_access_and_security.audit_log_required"],
                message="audit_log_required must be true only for HIPAA/PCI-DSS/SOC2; otherwise false.",
            )
        )

    # Failure tolerance should match redundancy intent.
    # If we are running >1 web/app VMs (or explicitly require a load balancer),
    # claiming we cannot tolerate a single server failure is contradictory.
    has_instance_redundancy = (b5.web_vm_count or 0) > 1 or (b5.app_vm_count or 0) > 1
    if (has_instance_redundancy or b2.load_balancer_required) and b5.single_server_failure_tolerance is False:
        v.append(
            ConsistencyViolation(
                rule="failure_tolerance_with_redundancy",
                fields=[
                    "analysis_block_2_architecture_pattern.load_balancer_required",
                    "analysis_block_5_availability.web_vm_count",
                    "analysis_block_5_availability.app_vm_count",
                    "analysis_block_5_availability.single_server_failure_tolerance",
                ],
                message=(
                    "single_server_failure_tolerance must be true when web_vm_count>1 or app_vm_count>1 "
                    "or load_balancer_required=true."
                ),
            )
        )

    # High SLA implies HA posture. Keep minimal but non-contradictory.
    if b5.sla_target == "99.99" and b5.multi_az is False:
        v.append(
            ConsistencyViolation(
                rule="multi_az_for_9999",
                fields=["analysis_block_5_availability.sla_target", "analysis_block_5_availability.multi_az"],
                message='sla_target="99.99" implies multi_az=true.',
            )
        )

    # Note: We intentionally do NOT enforce deterministic mappings between sla_target and:
    # - backup_policy
    # - db_replica
    # - multi_az
    # And we do NOT hard-restrict cost_optimised_variant_available.
    # These are architecture choices that the LLM (and user answers) should control.

    return v

