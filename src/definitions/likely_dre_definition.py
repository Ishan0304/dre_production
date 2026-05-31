"""Definition engine for computable likely drug-resistant epilepsy inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from core import EvidenceStatus
from definitions.evidence_model import LikelyDREResult, PatientEvidenceBundle


EPILEPSY_REASON = "epilepsy or recurrent seizure care evidence present"
ASM_REASON = "two or more distinct ASM schedules attempted"
SEIZURE_BURDEN_REASON = "persistent seizure burden present after escalation to second ASM"


@dataclass(slots=True)
class LikelyDREDefinition:
    """Evaluate patient evidence for a strict computable likely DRE inference.

    This engine is not a formal adjudicated ILAE DRE diagnosis. Tolerance,
    appropriateness, proper use, and sustained seizure freedom are often not
    fully ascertainable from routine patient data alone.
    """

    definition_version: str = "v1"

    def evaluate(self, bundle: PatientEvidenceBundle) -> LikelyDREResult:
        """Evaluate one patient evidence bundle against the likely DRE rule."""

        epilepsy_present = self._evaluate_epilepsy_evidence(bundle)
        asm_present = self._evaluate_asm_evidence(bundle)
        seizure_burden_present = self._evaluate_seizure_burden(bundle)

        reasons: list[str] = []
        if epilepsy_present:
            reasons.append(EPILEPSY_REASON)
        if asm_present:
            reasons.append(ASM_REASON)
        if seizure_burden_present:
            reasons.append(SEIZURE_BURDEN_REASON)

        likely_dre = epilepsy_present and asm_present and seizure_burden_present

        return LikelyDREResult(
            patient_id=bundle.patient_id,
            likely_dre=likely_dre,
            reasons=reasons,
            missing_evidence=self._collect_missing_evidence(bundle),
            evidence_completeness_score=bundle.completeness.completeness_score,
            definition_version=self.definition_version,
            notes=(
                "Computable inference only, not formal adjudicated ILAE DRE. "
                "Routine data may not fully establish tolerance, appropriateness, "
                "proper use, or sustained seizure freedom."
            ),
        )

    def evaluation_summary(self, bundle: PatientEvidenceBundle) -> dict[str, Any]:
        """Return an auditable summary of the component condition checks."""

        epilepsy_present = self._evaluate_epilepsy_evidence(bundle)
        asm_present = self._evaluate_asm_evidence(bundle)
        seizure_burden_present = self._evaluate_seizure_burden(bundle)

        return {
            "patient_id": bundle.patient_id,
            "definition_version": self.definition_version,
            "epilepsy_condition": epilepsy_present,
            "asm_condition": asm_present,
            "seizure_burden_condition": seizure_burden_present,
            "likely_dre": epilepsy_present and asm_present and seizure_burden_present,
            "missing_evidence": self._collect_missing_evidence(bundle),
        }

    @staticmethod
    def _evaluate_epilepsy_evidence(bundle: PatientEvidenceBundle) -> bool:
        evidence = bundle.epilepsy_evidence
        return evidence.has_epilepsy_diagnosis or evidence.has_recurrent_seizure_care

    @staticmethod
    def _evaluate_asm_evidence(bundle: PatientEvidenceBundle) -> bool:
        evidence = bundle.asm_treatment_evidence
        return evidence.has_two_or_more_distinct_asms and evidence.distinct_asm_count >= 2

    @staticmethod
    def _evaluate_seizure_burden(bundle: PatientEvidenceBundle) -> bool:
        evidence = bundle.seizure_burden_evidence
        return evidence.has_persistent_seizure_burden and evidence.post_second_asm_event_count >= 1

    def _collect_missing_evidence(self, bundle: PatientEvidenceBundle) -> list[str]:
        missing: list[str] = []

        if (
            bundle.epilepsy_evidence.status is EvidenceStatus.MISSING
            or not self._evaluate_epilepsy_evidence(bundle)
        ):
            missing.append("missing epilepsy evidence")

        if (
            bundle.asm_treatment_evidence.status is EvidenceStatus.MISSING
            or not self._evaluate_asm_evidence(bundle)
        ):
            missing.append("missing ASM treatment evidence")

        if (
            bundle.seizure_burden_evidence.status is EvidenceStatus.MISSING
            or not self._evaluate_seizure_burden(bundle)
        ):
            missing.append("missing seizure burden evidence")

        if bundle.asm_treatment_evidence.second_asm_start_time is None:
            missing.append("missing second ASM temporal anchor")

        missing.extend(bundle.completeness.missing_fields)
        return list(dict.fromkeys(missing))
