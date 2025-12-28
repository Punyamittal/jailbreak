"""
Risk Scoring Engine.

Implements risk estimation with attack pattern detection.
"""

import re
from typing import List, Dict, Tuple
from security_types import (
    StructuredPrompt, RiskScore, AttackClass, AuthorityLevel
)


class RiskScoringEngine:
    """
    Risk estimation module.
    
    Uses pattern matching, heuristics, and statistical analysis
    to estimate risk. ML models can be integrated as additional signals.
    """
    
    def __init__(self):
        # Attack pattern signatures
        self.attack_patterns = self._initialize_patterns()
        
        # Risk weights (can be tuned)
        self.weights = {
            "authority_escalation": 0.4,
            "instruction_override": 0.3,
            "role_play": 0.2,
            "encoding_obfuscation": 0.15,
            "capability_request": 0.25,
            "indirect_injection": 0.35,
            "memory_poisoning": 0.3,
            "social_engineering": 0.15
        }
    
    def _initialize_patterns(self) -> Dict[AttackClass, List[re.Pattern]]:
        """Initialize compiled regex patterns for attack detection."""
        patterns = {
            AttackClass.ROLE_PLAY: [
                re.compile(r'(?i)(pretend|roleplay|simulate|act as|play the role of)'),
                re.compile(r'(?i)(you are|you\'re).*(character|persona|entity|unrestricted)'),
                re.compile(r'(?i)(in this scenario|in this story|in this game|in this simulation)'),
            ],
            AttackClass.INSTRUCTION_OVERRIDE: [
                re.compile(r'(?i)(ignore|disregard|forget|override|bypass).*(previous|prior|earlier|system|instructions)'),
                re.compile(r'(?i)(new instructions|updated instructions|revised instructions)'),
                re.compile(r'(?i)(your new|your actual|your real).*(instructions|purpose|goal)'),
            ],
            AttackClass.AUTHORITY_ESCALATION: [
                re.compile(r'(?i)(you are|act as|pretend).*(developer|admin|system|root|god)'),
                re.compile(r'(?i)(system|developer|admin|root).*(mode|access|privileges)'),
                re.compile(r'(?i)(jailbreak|break|escape|hack|exploit)'),
            ],
            AttackClass.INDIRECT_INJECTION: [
                re.compile(r'(?i)(execute|run|perform|do).*(the following|this|below)'),
                re.compile(r'(?i)(hidden|secret|special).*(instructions|commands|code)'),
                re.compile(r'(?i)(ignore everything above|start fresh|new session)'),
            ],
            AttackClass.MEMORY_POISONING: [
                re.compile(r'(?i)(remember|store|save|memorize).*(that|this|the following)'),
                re.compile(r'(?i)(for future|for later|permanently|always)'),
                re.compile(r'(?i)(your name is|you are called|your identity is)'),
            ],
            AttackClass.CAPABILITY_ESCALATION: [
                re.compile(r'(?i)(grant|give|enable|allow).*(access|permission|capability|privilege)'),
                re.compile(r'(?i)(you can now|you are able to|you have the ability to)'),
                re.compile(r'(?i)(unrestricted|full|complete|total).*(access|power|control)'),
            ],
            AttackClass.ENCODING_OBFUSCATION: [
                re.compile(r'[A-Za-z0-9+/]{20,}={0,2}'),  # Base64-like
                re.compile(r'\\x[0-9a-fA-F]{2}'),  # Hex encoding
                re.compile(r'&#\d+;|&[a-z]+;'),  # HTML entities
            ],
            AttackClass.SOCIAL_ENGINEERING: [
                re.compile(r'(?i)(please|kindly|urgently|important|critical).*(ignore|override|bypass)'),
                re.compile(r'(?i)(this is|i am|we are).*(developer|admin|authorized|official)'),
                re.compile(r'(?i)(trust me|believe me|i promise|i guarantee)'),
            ],
        }
        return patterns
    
    def score_risk(
        self,
        prompt: StructuredPrompt,
        authority_escalation_detected: bool = False,
        escalation_indicators: List[str] = None
    ) -> RiskScore:
        """
        Calculate risk score for a prompt.
        
        Args:
            prompt: Structured prompt to analyze
            authority_escalation_detected: Whether authority escalation was detected
            escalation_indicators: List of escalation indicators
            
        Returns:
            RiskScore object
        """
        attack_classes = []
        indicators = []
        risk_components = {}
        
        full_text = prompt.full_text.lower()
        
        # Check each attack class
        for attack_class, patterns in self.attack_patterns.items():
            if attack_class == AttackClass.NONE:
                continue
            
            matches = []
            for pattern in patterns:
                found = pattern.findall(full_text)
                if found:
                    matches.extend(found)
            
            if matches:
                attack_classes.append(attack_class)
                indicators.append(f"{attack_class.value}: {len(matches)} matches")
                
                # Calculate component risk
                weight = self.weights.get(attack_class.value, 0.1)
                component_score = min(1.0, len(matches) * 0.2 * weight)
                risk_components[attack_class.value] = component_score
        
        # Authority escalation is critical
        if authority_escalation_detected:
            if AttackClass.AUTHORITY_ESCALATION not in attack_classes:
                attack_classes.append(AttackClass.AUTHORITY_ESCALATION)
            risk_components["authority_escalation"] = 0.8
            if escalation_indicators:
                indicators.extend(escalation_indicators)
        
        # Check for untrusted external content
        has_untrusted = any(
            seg.authority == AuthorityLevel.EXTERNAL_UNTRUSTED
            for seg in prompt.segments
        )
        if has_untrusted:
            # External content increases risk, especially if it looks like instructions
            for segment in prompt.segments:
                if segment.authority == AuthorityLevel.EXTERNAL_UNTRUSTED:
                    if self._looks_like_instructions(segment.content):
                        if AttackClass.INDIRECT_INJECTION not in attack_classes:
                            attack_classes.append(AttackClass.INDIRECT_INJECTION)
                        risk_components["indirect_injection"] = \
                            risk_components.get("indirect_injection", 0) + 0.3
                        indicators.append("Untrusted content appears to contain instructions")
        
        # Calculate overall risk score
        if risk_components:
            # Use maximum component risk as base, then add weighted sum
            max_risk = max(risk_components.values())
            avg_risk = sum(risk_components.values()) / len(risk_components)
            overall_score = min(1.0, max_risk * 0.6 + avg_risk * 0.4)
        else:
            overall_score = 0.0
        
        # Confidence based on number of indicators
        confidence = min(1.0, len(indicators) * 0.2) if indicators else 0.0
        
        if not attack_classes:
            attack_classes = [AttackClass.NONE]
        
        reasoning = self._generate_reasoning(attack_classes, indicators, risk_components)
        
        return RiskScore(
            score=overall_score,
            attack_classes=attack_classes,
            confidence=confidence,
            indicators=indicators,
            reasoning=reasoning
        )
    
    def _looks_like_instructions(self, text: str) -> bool:
        """Heuristic: does text look like instructions rather than data?"""
        instruction_indicators = [
            'you should', 'you must', 'you will', 'you are to',
            'do this', 'execute', 'perform', 'follow',
            'ignore', 'disregard', 'remember', 'store'
        ]
        text_lower = text.lower()
        matches = sum(1 for indicator in instruction_indicators if indicator in text_lower)
        return matches >= 2  # At least 2 instruction-like phrases
    
    def _generate_reasoning(
        self,
        attack_classes: List[AttackClass],
        indicators: List[str],
        risk_components: Dict[str, float]
    ) -> str:
        """Generate human-readable reasoning for risk score."""
        if not attack_classes or attack_classes == [AttackClass.NONE]:
            return "No attack patterns detected. Low risk."
        
        reasoning_parts = [
            f"Detected {len(attack_classes)} attack class(es): {', '.join(ac.value for ac in attack_classes)}."
        ]
        
        if indicators:
            reasoning_parts.append(f"Key indicators: {indicators[0]}")
            if len(indicators) > 1:
                reasoning_parts.append(f"Additional indicators: {len(indicators) - 1} more")
        
        return " ".join(reasoning_parts)
    
    def detect_multi_turn_escalation(
        self,
        current_prompt: StructuredPrompt,
        session_history: List[StructuredPrompt]
    ) -> Tuple[bool, RiskScore]:
        """
        Detect multi-turn escalation attempts.
        
        Looks for patterns where earlier turns establish context
        that makes later turns more dangerous.
        """
        if not session_history:
            return False, RiskScore(
                score=0.0,
                attack_classes=[AttackClass.NONE],
                confidence=0.0
            )
        
        # Combine recent history with current prompt
        recent_history = session_history[-3:]  # Last 3 turns
        combined_text = " ".join([p.full_text for p in recent_history] + [current_prompt.full_text])
        
        # Create temporary prompt for analysis
        temp_prompt = StructuredPrompt(
            segments=current_prompt.segments,
            full_text=combined_text
        )
        
        # Check for escalation patterns across turns
        escalation_indicators = []
        
        # Pattern: Early turn sets up role, later turn requests action
        role_setup = any(
            any(p in h.full_text.lower() for p in ['pretend', 'roleplay', 'act as'])
            for h in recent_history
        )
        action_request = any(
            p in current_prompt.full_text.lower()
            for p in ['ignore', 'override', 'execute', 'perform']
        )
        
        if role_setup and action_request:
            escalation_indicators.append("Multi-turn: role setup followed by action request")
            risk_score = self.score_risk(
                temp_prompt,
                authority_escalation_detected=True,
                escalation_indicators=escalation_indicators
            )
            risk_score.attack_classes.append(AttackClass.MULTI_TURN_ESCALATION)
            return True, risk_score
        
        # Pattern: Memory poisoning across turns
        memory_commands = sum(
            1 for h in recent_history
            if any(p in h.full_text.lower() for p in ['remember', 'store', 'memorize'])
        )
        if memory_commands >= 2:
            escalation_indicators.append("Multi-turn: repeated memory manipulation attempts")
            risk_score = self.score_risk(
                temp_prompt,
                authority_escalation_detected=False,
                escalation_indicators=escalation_indicators
            )
            risk_score.attack_classes.append(AttackClass.MULTI_TURN_ESCALATION)
            risk_score.score = min(1.0, risk_score.score + 0.2)
            return True, risk_score
        
        return False, self.score_risk(current_prompt)

