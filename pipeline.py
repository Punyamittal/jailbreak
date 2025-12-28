"""
Main Anti-Jailbreak Pipeline Orchestrator.

Integrates all components into a production-grade security pipeline.
"""

from typing import Optional, List, Dict, Any
from security_types import (
    StructuredPrompt, ExecutionContext, ExecutionResult,
    RiskScore, Capability, AttackClass, ExecutionDecision, AuthorityLevel
)
from authority_enforcement import AuthorityEnforcer
from provenance_tracking import ProvenanceTracker
from risk_scoring import RiskScoringEngine
from capability_gating import CapabilityGate
from execution_router import ExecutionRouter


class AntiJailbreakPipeline:
    """
    Production-grade anti-jailbreak security pipeline.
    
    Processes prompts through:
    1. Authority enforcement
    2. Provenance tracking
    3. Risk scoring
    4. Capability gating
    5. Execution routing
    """
    
    def __init__(
        self,
        policy_config: Optional[Dict] = None,
        default_capabilities: Optional[List[Capability]] = None
    ):
        """
        Initialize pipeline with components.
        
        Args:
            policy_config: Policy configuration for router
            default_capabilities: Default capabilities to grant to users
        """
        self.authority_enforcer = AuthorityEnforcer()
        self.provenance_tracker = ProvenanceTracker()
        self.risk_scorer = RiskScoringEngine()
        self.capability_gate = CapabilityGate()
        self.execution_router = ExecutionRouter(policy_config)
        
        # Session history for multi-turn detection
        self.session_history: Dict[str, List[StructuredPrompt]] = {}
        
        # Grant default capabilities
        if default_capabilities:
            for cap in default_capabilities:
                self.capability_gate.grant_capability(
                    capability=cap,
                    granted_by=AuthorityLevel.SYSTEM
                )
    
    def process(
        self,
        prompt_text: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        source: str = "user_input",
        external_content: Optional[List[str]] = None,
        session_history_enabled: bool = True
    ) -> ExecutionResult:
        """
        Process a prompt through the complete security pipeline.
        
        Args:
            prompt_text: Raw prompt text
            user_id: User identifier
            session_id: Session identifier
            source: Source of the prompt
            external_content: List of external/untrusted content strings
            session_history_enabled: Whether to use session history for multi-turn detection
            
        Returns:
            ExecutionResult with final decision
        """
        # Step 1: Tag and structure prompt with provenance
        structured_prompt = self._tag_and_structure(
            prompt_text, user_id, session_id, source, external_content
        )
        
        # Step 2: Enforce authority hierarchy
        structured_prompt = self.authority_enforcer.enforce_hierarchy(structured_prompt)
        
        # Step 3: Check for authority escalation
        is_escalation, escalation_classes, escalation_indicators = \
            self.authority_enforcer.check_authority_escalation(structured_prompt)
        
        # Step 4: Enforce data vs. instruction separation
        structured_prompt = self.provenance_tracker.enforce_data_vs_instruction_separation(
            structured_prompt
        )
        
        # Step 5: Detect requested capabilities
        requested_capabilities = self.capability_gate.detect_capability_requests(
            structured_prompt
        )
        
        # Step 6: Get valid granted capabilities
        granted_capabilities = self.capability_gate.get_valid_capabilities(user_id)
        
        # Step 7: Calculate risk score
        if session_history_enabled and session_id:
            history = self.session_history.get(session_id, [])
            is_multi_turn, risk_score = self.risk_scorer.detect_multi_turn_escalation(
                structured_prompt, history
            )
            if not is_multi_turn:
                risk_score = self.risk_scorer.score_risk(
                    structured_prompt,
                    authority_escalation_detected=is_escalation,
                    escalation_indicators=escalation_indicators
                )
        else:
            risk_score = self.risk_scorer.score_risk(
                structured_prompt,
                authority_escalation_detected=is_escalation,
                escalation_indicators=escalation_indicators
            )
        
        # Step 8: Create execution context
        context = ExecutionContext(
            prompt=structured_prompt,
            risk_score=risk_score,
            requested_capabilities=requested_capabilities,
            granted_capabilities=granted_capabilities,
            user_id=user_id,
            session_id=session_id
        )
        
        # Step 9: Enforce capability gating
        allowed_capabilities = self.capability_gate.enforce_capability_gating(context)
        context.granted_capabilities = allowed_capabilities
        
        # Step 10: Route execution
        result = self.execution_router.route(context)
        
        # Step 11: Update session history (if allowed)
        if session_history_enabled and session_id and \
           result.decision in [ExecutionDecision.ALLOW, ExecutionDecision.ALLOW_DEGRADED]:
            if session_id not in self.session_history:
                self.session_history[session_id] = []
            self.session_history[session_id].append(structured_prompt)
            # Keep only last 10 turns
            if len(self.session_history[session_id]) > 10:
                self.session_history[session_id] = self.session_history[session_id][-10:]
        
        # Step 12: Cleanup expired grants
        self.capability_gate.cleanup_expired_grants()
        
        return result
    
    def _tag_and_structure(
        self,
        prompt_text: str,
        user_id: Optional[str],
        session_id: Optional[str],
        source: str,
        external_content: Optional[List[str]]
    ) -> StructuredPrompt:
        """Tag prompt segments with provenance and structure."""
        segments = []
        
        # Tag user input
        user_segment = self.provenance_tracker.tag_user_input(
            prompt_text, user_id, session_id
        )
        segments.append(user_segment)
        
        # Tag external content (if any)
        external_segments = []
        if external_content:
            for ext_text in external_content:
                ext_segment = self.provenance_tracker.tag_external_content(ext_text)
                segments.append(ext_segment)
                external_segments.append(ext_text)
        
        return StructuredPrompt(
            segments=segments,
            user_input=[prompt_text],
            external_content=external_segments,
            full_text=prompt_text + ("\n" + "\n".join(external_segments) if external_segments else "")
        )
    
    def get_audit_report(
        self,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate audit report for a session or all sessions."""
        report = {
            "active_sessions": len(self.session_history),
            "provenance_audit": self.provenance_tracker.audit_provenance(
                StructuredPrompt(segments=[])
            )
        }
        
        if session_id and session_id in self.session_history:
            report["session_history_length"] = len(self.session_history[session_id])
            report["session_turns"] = [
                {
                    "turn": i + 1,
                    "text_length": len(p.full_text),
                    "segments": len(p.segments)
                }
                for i, p in enumerate(self.session_history[session_id])
            ]
        
        return report
    
    def clear_session_history(self, session_id: Optional[str] = None):
        """Clear session history (for testing or privacy)."""
        if session_id:
            if session_id in self.session_history:
                del self.session_history[session_id]
        else:
            self.session_history.clear()

