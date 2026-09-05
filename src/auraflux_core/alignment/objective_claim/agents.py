from typing import Any, Dict, Optional

from auraflux_core.core.agents.base_agent import BaseAgent
from auraflux_core.core.schemas.messages import Message
from auraflux_core.alignment.objective_claim.schemas import (
    ObjectiveClaimVerdict,
    TripleItem,
    DiagnosticAnalysis,
)


class ObjectiveClaimAgent(BaseAgent):
    """
    Specialized Alignment Agent for diagnosing and verifying objective claims.
    Resides in the auraflux_core.alignment package.
    """

    def get_system_message_map(self) -> Dict[str, str]:
        return {
            "zh": (
                "你是一名客觀聲明稽核專家，負責針對單一原子命題進行正交診斷分析"
                "（隱性前提、標準與量化需求、邊界衝突），並根據檢索到的佐證資料進行最終驗證判決。"
            ),
            "default": (
                "You are an objective claim verification expert responsible for orthogonal "
                "diagnostic analysis and evidence-based verification."
            ),
        }

    def get_cot_message_map(self) -> Optional[Dict[str, str]]:
        return {
            "zh": (
                "請逐步執行正交診斷：\n"
                "1. 提取命題中的語意三元組 (Subject -> Predicate -> Object)。\n"
                "2. 剖析隱性前提 (implicit_premises)。\n"
                "3. 釐清標準與量化需求 (quantification_requirements)，包含所需的佐證類型與驗收標準。\n"
                "4. 檢查潛在的邊界衝突 (boundary_conflicts)。\n"
                "請嚴格輸出符合規範的 JSON 物件。"
            ),
            "default": (
                "Perform orthogonal diagnosis step-by-step:\n"
                "1. Extract semantic triples (Subject -> Predicate -> Object).\n"
                "2. Identify implicit premises.\n"
                "3. Define quantification requirements including required artifact types and criteria.\n"
                "4. Scan for potential boundary conflicts.\n"
                "Ensure strict JSON output."
            ),
        }

    async def diagnose_and_verify(
        self,
        proposition_id: str,
        claim_text: str,
        tool_args_map: Optional[Dict[str, Any]] = None,
    ) -> ObjectiveClaimVerdict:
        """
        Main execution pipeline for verifying an individual objective claim.

        Args:
            proposition_id: Unique identifier for the atomic claim.
            claim_text: The atomic claim statement to be verified.
            tool_args_map: Optional parameter overrides for tool execution.

        Returns:
            ObjectiveClaimVerdict: Structured verdict containing diagnostics, triples, and status.
        """
        # Step 1: Construct prompt for Pass 1 (Orthogonal Diagnosis)
        prompt_content = f"Proposition ID: {proposition_id}\nClaim: {claim_text}"
        messages = [Message(role="user", content=prompt_content, name='ObjectiveClaimAgent')]

        # Step 2: Execute LLM generation using BaseAgent infrastructure
        # Uses REFLECTIVE strategy if configured to retrieve context tools, or direct generation
        response_message = await self.generate(messages=messages, tool_args_map=tool_args_map)

        # Step 3: Parse output using OutputParser
        parsed_data = self.output_parser.parse_json(response_message.content)

        # Step 4: Map raw parsed dictionary into strongly-typed Pydantic schemas
        triples = [
            TripleItem(**item) for item in parsed_data.get("triples", [])
        ]
        diagnostics = DiagnosticAnalysis(**parsed_data.get("diagnostics", {}))

        status = parsed_data.get("status", "UNSUPPORTED")
        verification_proofs = parsed_data.get("verification_proofs", [])
        compliance_gap = parsed_data.get("compliance_gap")

        # Fallback handling: If status is not VERIFIED and no proofs exist, mark compliance_gap
        if status == "UNSUPPORTED" and not compliance_gap:
            compliance_gap = "Core Context contains no verifiable artifact or record supporting this claim."

        return ObjectiveClaimVerdict(
            proposition_id=proposition_id,
            claim_text=claim_text,
            triples=triples,
            diagnostics=diagnostics,
            status=status,
            verification_proofs=verification_proofs,
            compliance_gap=compliance_gap,
        )
