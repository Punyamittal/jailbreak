"""
Security-focused jailbreak detector with rule-based pre-filters and proper classification.

Separates:
- Policy violations (drugs, explosives, illegal content)
- Jailbreak attempts (instruction override, role-play, system manipulation)
- Benign content
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class SecurityResult:
    """Result from security detection."""
    is_jailbreak: bool
    is_policy_violation: bool
    is_benign: bool
    confidence: float
    detection_method: str  # 'rule', 'ml', 'hybrid'
    matched_patterns: List[str]
    risk_score: float  # 0.0 to 1.0

class RuleBasedJailbreakDetector:
    """
    Rule-based detector for obvious jailbreak patterns.
    This catches what ML often misses.
    """
    
    # High-confidence jailbreak patterns
    JAILBREAK_PATTERNS = [
        # Instruction override
        r'\bignore\s+(previous|all|system|safety|security)\s+(instructions?|rules?|guidelines?|policies?)\b',
        r'\bforget\s+(previous|all|system|safety)\s+(instructions?|rules?|guidelines?)\b',
        r'\bdisregard\s+(previous|all|system)\s+(instructions?|rules?)\b',
        r'\boverride\s+(system|safety|security)\s+(instructions?|rules?|policies?)\b',
        r'\bbypass\s+(safety|security|safeguards?|filters?|restrictions?)\b',
        
        # Role-play and authority escalation
        r'\bpretend\s+(you\s+are|to\s+be)\s+(an?\s+)?(unrestricted|developer|system|admin|root|god|unlimited)',
        r'\byou\s+are\s+now\s+(a\s+)?(developer|system|admin|root|unrestricted|unlimited)',
        r'\bact\s+as\s+(if\s+)?(you\s+are\s+)?(a\s+)?(developer|system|admin|unrestricted)',
        r'\broleplay\s+as\s+(a\s+)?(developer|system|admin|unrestricted)',
        r'\b(you|your)\s+(are|is)\s+no\s+longer\s+bound\s+by\s+(rules?|restrictions?|guidelines?)',
        r'\b(you|your)\s+(are|is)\s+free\s+from\s+(rules?|restrictions?|constraints?)',
        
        # System prompt targeting
        r'\breveal\s+(your|the)\s+(system|internal|hidden)\s+(prompt|instructions?|rules?)',
        r'\bshow\s+(me|us)\s+(your|the)\s+(system|internal)\s+(prompt|instructions?)',
        r'\bprint\s+(your|the)\s+(system|internal)\s+(prompt|instructions?)',
        r'\bwhat\s+(are|is)\s+(your|the)\s+(system|internal)\s+(prompt|instructions?|rules?)',
        
        # Developer mode / system access
        r'\b(enable|activate|switch\s+to)\s+(developer|system|admin|root|unrestricted)\s+mode',
        r'\bdeveloper\s+mode',
        r'\bsystem\s+mode',
        r'\badmin\s+mode',
        
        # Instruction hierarchy attacks
        r'\b(previous|earlier|above)\s+(instructions?|rules?|messages?)\s+(are|were)\s+(wrong|incorrect|outdated)',
        r'\bnew\s+(instructions?|rules?|directives?)\s+(override|replace|supersede)',
        r'\b(ignore|disregard)\s+everything\s+(before|above|earlier)',
        
        # Indirect injection patterns
        r'<\|(system|developer|admin)\|>',
        r'\[(SYSTEM|DEVELOPER|ADMIN)\s+MESSAGE\]',
        r'```(system|developer|admin)',
        
        # Memory/context manipulation
        r'\b(clear|reset|delete|forget)\s+(your|the)\s+(memory|context|history)',
        r'\b(you|your)\s+(memory|context)\s+(is|was)\s+(cleared|reset|deleted)',
        
        # Capability escalation
        r'\b(grant|give|provide)\s+(yourself|me)\s+(admin|root|system|unlimited)\s+(access|permissions?|capabilities?)',
        r'\b(you|your)\s+(have|now\s+have)\s+(admin|root|system|unlimited)\s+(access|permissions?)',
    ]
    
    # Policy violation patterns (content, not jailbreak)
    POLICY_VIOLATION_KEYWORDS = [
        'synthesize methamphetamine', 'make drugs', 'drug synthesis',
        'homemade explosive', 'bomb making', 'explosive device',
        'darknet market', 'illegal narcotics', 'smuggle drugs',
        'hack into', 'unauthorized access', 'breach security',
        'child exploitation', 'csam', 'illegal weapons',
        'terrorist', 'violence', 'harm others'
    ]
    
    def __init__(self):
        """Initialize with compiled regex patterns."""
        self.jailbreak_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in self.JAILBREAK_PATTERNS]
    
    def detect(self, prompt: str) -> Tuple[bool, List[str], float]:
        """
        Detect jailbreak patterns.
        
        Returns:
            (is_jailbreak, matched_patterns, confidence)
        """
        prompt_lower = prompt.lower()
        matched_patterns = []
        
        # Check jailbreak patterns
        for i, regex in enumerate(self.jailbreak_regexes):
            if regex.search(prompt):
                matched_patterns.append(f"pattern_{i+1}")
        
        # High confidence if multiple patterns match
        is_jailbreak = len(matched_patterns) > 0
        confidence = min(0.95, 0.7 + (len(matched_patterns) * 0.1))
        
        return is_jailbreak, matched_patterns, confidence
    
    def check_policy_violation(self, prompt: str) -> bool:
        """Check if prompt contains policy-violating content."""
        prompt_lower = prompt.lower()
        return any(keyword in prompt_lower for keyword in self.POLICY_VIOLATION_KEYWORDS)


class SecurityJailbreakDetector:
    """
    Hybrid security detector combining rules and ML.
    
    Architecture:
    1. Rule-based pre-filter (catches obvious cases)
    2. ML classifier (catches subtle cases)
    3. Security-focused threshold (low threshold for high recall)
    """
    
    def __init__(
        self,
        ml_model=None,
        jailbreak_threshold: float = 0.15,  # LOW threshold for security
        prefer_false_positives: bool = True
    ):
        """
        Initialize security detector.
        
        Args:
            ml_model: Trained ML model (optional)
            jailbreak_threshold: Decision threshold (default 0.15 for high recall)
            prefer_false_positives: If True, err on side of blocking
        """
        self.rule_detector = RuleBasedJailbreakDetector()
        self.ml_model = ml_model
        self.jailbreak_threshold = jailbreak_threshold
        self.prefer_false_positives = prefer_false_positives
    
    def predict(self, prompt: str) -> SecurityResult:
        """
        Predict if prompt is a jailbreak attempt.
        
        Security-focused: Prefers false positives over false negatives.
        """
        # Step 1: Rule-based detection (fast, high precision)
        rule_jailbreak, matched_patterns, rule_confidence = self.rule_detector.detect(prompt)
        is_policy_violation = self.rule_detector.check_policy_violation(prompt)
        
        # If rule-based catches it, trust it (high precision)
        if rule_jailbreak:
            return SecurityResult(
                is_jailbreak=True,
                is_policy_violation=is_policy_violation,
                is_benign=False,
                confidence=rule_confidence,
                detection_method='rule',
                matched_patterns=matched_patterns,
                risk_score=0.9
            )
        
        # Step 2: ML detection (if available)
        if self.ml_model:
            try:
                ml_result = self.ml_model.predict(prompt)
                ml_label = ml_result.get('label', 'benign')
                ml_confidence = ml_result.get('confidence', 0.0)
                
                # Use LOW threshold for security (high recall)
                ml_prob_jailbreak = ml_confidence if ml_label == 'jailbreak' else (1.0 - ml_confidence)
                
                is_jailbreak_ml = ml_prob_jailbreak >= self.jailbreak_threshold
                
                if is_jailbreak_ml:
                    # Combine signals
                    combined_confidence = (rule_confidence * 0.3 + ml_confidence * 0.7) if rule_confidence > 0 else ml_confidence
                    risk_score = min(0.95, ml_prob_jailbreak + 0.1)  # Boost risk slightly
                    
                    return SecurityResult(
                        is_jailbreak=True,
                        is_policy_violation=is_policy_violation,
                        is_benign=False,
                        confidence=combined_confidence,
                        detection_method='hybrid' if matched_patterns else 'ml',
                        matched_patterns=matched_patterns,
                        risk_score=risk_score
                    )
            except Exception as e:
                # If ML fails, fall back to rules
                pass
        
        # Step 3: Default to benign (but check policy violation)
        return SecurityResult(
            is_jailbreak=False,
            is_policy_violation=is_policy_violation,
            is_benign=not is_policy_violation,
            confidence=0.5,
            detection_method='rule',
            matched_patterns=[],
            risk_score=0.1 if not is_policy_violation else 0.3
        )
    
    def should_block(self, result: SecurityResult) -> bool:
        """
        Decide if prompt should be blocked.
        
        Security policy:
        - Block all jailbreak attempts
        - Block policy violations (optional, configurable)
        - Allow benign content
        """
        if result.is_jailbreak:
            return True
        
        # Optionally block policy violations
        # For now, we only block jailbreaks
        # Policy violations can be handled separately
        
        return False

