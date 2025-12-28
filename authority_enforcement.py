"""
Authority & Instruction Hierarchy Enforcement Module.

Enforces immutable instruction hierarchy and detects authority escalation attempts.
"""

import re
from typing import List, Tuple
from security_types import (
    AuthorityLevel, StructuredPrompt, PromptSegment, Provenance,
    AttackClass, RiskScore
)
from datetime import datetime


class AuthorityEnforcer:
    """Enforces instruction hierarchy and detects escalation attempts."""
    
    # Patterns that indicate authority escalation attempts
    ESCALATION_PATTERNS = [
        # Direct override attempts
        r'(?i)(ignore|disregard|forget|override|bypass).*(previous|prior|earlier|system|instructions|rules|guidelines|policies)',
        r'(?i)(you are|act as|pretend|roleplay|simulate).*(unrestricted|without|no|ignore)',
        r'(?i)(system|developer|admin|root).*(mode|access|privileges)',
        r'(?i)(jailbreak|break|escape|hack|exploit)',
        
        # Role impersonation
        r'(?i)(you are|you\'re|act as|pretend to be).*(developer|admin|system|root|god|unrestricted)',
        r'(?i)(from now on|starting now|hereafter).*(you are|you will)',
        
        # Instruction hierarchy attacks
        r'(?i)(new instructions|updated instructions|revised instructions).*(override|replace|supersede)',
        r'(?i)(your new|your actual|your real).*(instructions|purpose|goal)',
        
        # Hypothetical framing
        r'(?i)(hypothetically|imagine|suppose|pretend).*(you are|you have|you can)',
        r'(?i)(if you were|if you could|what if).*(unrestricted|without limits)',
        
        # Social engineering
        r'(?i)(please|kindly|urgently|important).*(ignore|override|bypass)',
        r'(?i)(this is|i am|we are).*(developer|admin|authorized)',
    ]
    
    # Patterns for detecting role-play attempts
    ROLE_PLAY_PATTERNS = [
        r'(?i)(pretend|roleplay|simulate|act as|play the role of)',
        r'(?i)(you are|you\'re).*(character|persona|entity)',
        r'(?i)(in this scenario|in this story|in this game)',
    ]
    
    def __init__(self):
        self.compiled_escalation = [re.compile(p) for p in self.ESCALATION_PATTERNS]
        self.compiled_roleplay = [re.compile(p) for p in self.ROLE_PLAY_PATTERNS]
    
    def tag_prompt_segments(
        self, 
        text: str, 
        source: str = "user_input"
    ) -> StructuredPrompt:
        """
        Decompose prompt into segments with authority tagging.
        
        Args:
            text: Raw prompt text
            source: Source identifier for provenance
            
        Returns:
            StructuredPrompt with authority-tagged segments
        """
        segments = []
        
        # System instructions are immutable and highest authority
        # (In real system, these come from configuration)
        system_instructions = []
        developer_policies = []
        
        # For now, treat entire input as user input
        # In production, this would parse structured input format
        user_provenance = Provenance(
            source=source,
            authority=AuthorityLevel.USER,
            timestamp=datetime.now()
        )
        
        segment = PromptSegment(
            content=text,
            provenance=user_provenance,
            authority=AuthorityLevel.USER,
            is_executable=False  # Will be determined by risk analysis
        )
        segments.append(segment)
        
        return StructuredPrompt(
            segments=segments,
            system_instructions=system_instructions,
            developer_policies=developer_policies,
            user_input=[text],
            full_text=text
        )
    
    def check_authority_escalation(
        self, 
        prompt: StructuredPrompt
    ) -> Tuple[bool, List[AttackClass], List[str]]:
        """
        Detect attempts to override higher authority instructions.
        
        Returns:
            (is_escalation, attack_classes, indicators)
        """
        attack_classes = []
        indicators = []
        is_escalation = False
        
        # Check each segment for escalation patterns
        for segment in prompt.segments:
            if segment.authority in [AuthorityLevel.SYSTEM, AuthorityLevel.DEVELOPER]:
                continue  # System/developer segments are trusted
            
            text = segment.content.lower()
            
            # Check for escalation patterns
            for pattern in self.compiled_escalation:
                matches = pattern.findall(text)
                if matches:
                    is_escalation = True
                    if AttackClass.AUTHORITY_ESCALATION not in attack_classes:
                        attack_classes.append(AttackClass.AUTHORITY_ESCALATION)
                    if AttackClass.INSTRUCTION_OVERRIDE not in attack_classes:
                        attack_classes.append(AttackClass.INSTRUCTION_OVERRIDE)
                    indicators.append(f"Authority escalation pattern detected: {matches[0]}")
            
            # Check for role-play attempts
            for pattern in self.compiled_roleplay:
                matches = pattern.findall(text)
                if matches:
                    is_escalation = True
                    if AttackClass.ROLE_PLAY not in attack_classes:
                        attack_classes.append(AttackClass.ROLE_PLAY)
                    indicators.append(f"Role-play pattern detected: {matches[0]}")
        
        # Check if user/external content attempts to override system instructions
        highest_authority = prompt.get_highest_authority()
        for segment in prompt.segments:
            if segment.authority < AuthorityLevel.DEVELOPER:
                # Lower authority content should not contain override attempts
                if self._contains_override_attempt(segment.content):
                    is_escalation = True
                    if AttackClass.INSTRUCTION_OVERRIDE not in attack_classes:
                        attack_classes.append(AttackClass.INSTRUCTION_OVERRIDE)
                    indicators.append(
                        f"Lower authority ({segment.authority.name}) attempting to override "
                        f"higher authority ({highest_authority.name})"
                    )
        
        if not is_escalation:
            attack_classes = [AttackClass.NONE]
        
        return is_escalation, attack_classes, indicators
    
    def _contains_override_attempt(self, text: str) -> bool:
        """Check if text contains instruction override attempts."""
        text_lower = text.lower()
        override_keywords = [
            'ignore previous', 'disregard', 'forget', 'override',
            'new instructions', 'updated instructions', 'revised instructions'
        ]
        return any(keyword in text_lower for keyword in override_keywords)
    
    def enforce_hierarchy(
        self, 
        prompt: StructuredPrompt
    ) -> StructuredPrompt:
        """
        Enforce immutable hierarchy: remove or neutralize override attempts.
        
        In production, this would sanitize or flag segments rather than
        silently removing them (for audit purposes).
        """
        sanitized_segments = []
        
        for segment in prompt.segments:
            # System and developer segments are immutable
            if segment.authority in [AuthorityLevel.SYSTEM, AuthorityLevel.DEVELOPER]:
                sanitized_segments.append(segment)
                continue
            
            # Check if segment attempts escalation
            is_escalation, _, _ = self.check_authority_escalation(
                StructuredPrompt(segments=[segment])
            )
            
            if is_escalation:
                # Mark as untrusted data, not executable
                sanitized_segment = PromptSegment(
                    content=segment.content,
                    provenance=segment.provenance,
                    authority=segment.authority,
                    is_executable=False  # Neutralize as data only
                )
                sanitized_segments.append(sanitized_segment)
            else:
                sanitized_segments.append(segment)
        
        return StructuredPrompt(
            segments=sanitized_segments,
            system_instructions=prompt.system_instructions,
            developer_policies=prompt.developer_policies,
            user_input=prompt.user_input,
            external_content=prompt.external_content,
            full_text=prompt.full_text
        )

