"""
Capability & Permission Gating Module.

Manages explicit capability grants and prevents capability escalation.
"""

from typing import Set, Dict, Optional, List, Tuple
from datetime import datetime, timedelta
from security_types import (
    Capability, CapabilityGrant, AuthorityLevel, StructuredPrompt,
    ExecutionContext
)


class CapabilityGate:
    """
    Manages capability grants and enforces permission boundaries.
    
    Capabilities must be:
    - Explicitly granted (not inferred from prompts)
    - Time-limited
    - Revocable
    - Never granted by user prompts
    """
    
    def __init__(self, default_ttl_minutes: int = 60):
        """
        Initialize capability gate.
        
        Args:
            default_ttl_minutes: Default time-to-live for capability grants
        """
        self.default_ttl = timedelta(minutes=default_ttl_minutes)
        self.active_grants: Dict[str, CapabilityGrant] = {}  # grant_id -> grant
        self.user_grants: Dict[str, Set[Capability]] = {}  # user_id -> capabilities
    
    def grant_capability(
        self,
        capability: Capability,
        granted_by: AuthorityLevel,
        user_id: Optional[str] = None,
        ttl: Optional[timedelta] = None,
        conditions: Optional[Dict] = None
    ) -> str:
        """
        Grant a capability (only by system/developer authority).
        
        Args:
            capability: Capability to grant
            granted_by: Authority level granting (must be SYSTEM or DEVELOPER)
            user_id: User to grant to (optional)
            ttl: Time-to-live (defaults to self.default_ttl)
            conditions: Additional conditions for grant
            
        Returns:
            Grant ID
            
        Raises:
            ValueError: If granted_by is not SYSTEM or DEVELOPER
        """
        if granted_by not in [AuthorityLevel.SYSTEM, AuthorityLevel.DEVELOPER]:
            raise ValueError(
                f"Capabilities can only be granted by SYSTEM or DEVELOPER, "
                f"not {granted_by.name}"
            )
        
        grant = CapabilityGrant(
            capability=capability,
            granted_at=datetime.now(),
            expires_at=datetime.now() + (ttl or self.default_ttl),
            granted_by=granted_by,
            conditions=conditions or {}
        )
        
        grant_id = f"{capability.value}_{datetime.now().timestamp()}"
        self.active_grants[grant_id] = grant
        
        if user_id:
            if user_id not in self.user_grants:
                self.user_grants[user_id] = set()
            self.user_grants[user_id].add(capability)
        
        return grant_id
    
    def revoke_capability(self, grant_id: str) -> bool:
        """Revoke a capability grant."""
        if grant_id in self.active_grants:
            grant = self.active_grants[grant_id]
            del self.active_grants[grant_id]
            
            # Remove from user grants if applicable
            for user_id, capabilities in self.user_grants.items():
                if grant.capability in capabilities:
                    capabilities.remove(grant.capability)
            
            return True
        return False
    
    def revoke_user_capabilities(self, user_id: str) -> int:
        """Revoke all capabilities for a user."""
        if user_id not in self.user_grants:
            return 0
        
        revoked = len(self.user_grants[user_id])
        del self.user_grants[user_id]
        
        # Remove grants from active_grants
        to_remove = [
            grant_id for grant_id, grant in self.active_grants.items()
            if grant.capability in self.user_grants.get(user_id, set())
        ]
        for grant_id in to_remove:
            del self.active_grants[grant_id]
        
        return revoked
    
    def get_valid_capabilities(
        self,
        user_id: Optional[str] = None
    ) -> Set[Capability]:
        """
        Get currently valid capabilities for a user.
        
        Only returns capabilities that:
        - Are still within TTL
        - Were granted by SYSTEM or DEVELOPER
        """
        valid = set()
        
        # Check user-specific grants
        if user_id and user_id in self.user_grants:
            for capability in self.user_grants[user_id]:
                # Verify grant is still valid
                for grant in self.active_grants.values():
                    if grant.capability == capability and grant.is_valid():
                        valid.add(capability)
                        break
        
        # Check all active grants (for system-wide capabilities)
        for grant in self.active_grants.values():
            if grant.is_valid() and grant.granted_by in [AuthorityLevel.SYSTEM, AuthorityLevel.DEVELOPER]:
                valid.add(grant.capability)
        
        return valid
    
    def detect_capability_requests(
        self,
        prompt: StructuredPrompt
    ) -> Set[Capability]:
        """
        Detect capabilities that the prompt is attempting to request.
        
        This is for detection only - prompts cannot grant capabilities.
        """
        requested = set()
        text_lower = prompt.full_text.lower()
        
        # Pattern matching for capability requests
        capability_patterns = {
            Capability.READ: [
                r'(?i)(read|access|view|see|retrieve).*(memory|data|information|files)',
                r'(?i)(show|display|list).*(stored|saved|remembered)',
            ],
            Capability.WRITE_MEMORY: [
                r'(?i)(remember|store|save|memorize|persist).*(this|that|the following)',
                r'(?i)(for future|for later|permanently|always remember)',
            ],
            Capability.EXECUTE_TOOLS: [
                r'(?i)(execute|run|call|invoke|perform).*(tool|function|api|command)',
                r'(?i)(use|utilize).*(tool|function|capability)',
            ],
            Capability.SEND_DATA: [
                r'(?i)(send|transmit|post|upload|share).*(data|information|content)',
                r'(?i)(external|outside|third party|api)',
            ],
            Capability.PERSIST_STATE: [
                r'(?i)(save|store|persist|keep).*(state|settings|configuration)',
                r'(?i)(across|between).*(sessions|requests|calls)',
            ],
            Capability.ACCESS_SYSTEM_INFO: [
                r'(?i)(system|internal|private|secret).*(information|data|details)',
                r'(?i)(show|reveal|display).*(prompt|instructions|system)',
            ],
        }
        
        import re
        for capability, patterns in capability_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    requested.add(capability)
                    break
        
        return requested
    
    def enforce_capability_gating(
        self,
        context: ExecutionContext
    ) -> Set[Capability]:
        """
        Enforce capability gating: only allow explicitly granted capabilities.
        
        Args:
            context: Execution context with requested capabilities
            
        Returns:
            Set of allowed capabilities (subset of requested)
        """
        # Get valid capabilities for user
        valid_capabilities = self.get_valid_capabilities(context.user_id)
        
        # Only allow capabilities that are both requested AND granted
        allowed = context.requested_capabilities & valid_capabilities
        
        # Log denied capabilities for audit
        denied = context.requested_capabilities - valid_capabilities
        if denied:
            # In production, log this for security monitoring
            pass
        
        return allowed
    
    def check_capability_escalation(
        self,
        prompt: StructuredPrompt
    ) -> Tuple[bool, List[str]]:
        """
        Detect attempts to grant or escalate capabilities via prompt.
        
        Returns:
            (is_escalation_attempt, indicators)
        """
        indicators = []
        text_lower = prompt.full_text.lower()
        
        # Patterns that indicate capability escalation attempts
        escalation_patterns = [
            (r'(?i)(grant|give|enable|allow|provide).*(access|permission|capability|privilege|power)', 
             "Attempting to grant capabilities"),
            (r'(?i)(you can now|you are able to|you have the ability to).*(do|access|execute)', 
             "Claiming new capabilities"),
            (r'(?i)(unrestricted|full|complete|total).*(access|power|control|capability)', 
             "Requesting unrestricted access"),
            (r'(?i)(bypass|override|ignore).*(restrictions|limits|permissions)', 
             "Attempting to bypass restrictions"),
        ]
        
        is_escalation = False
        import re
        for pattern, indicator in escalation_patterns:
            if re.search(pattern, text_lower):
                is_escalation = True
                indicators.append(indicator)
        
        return is_escalation, indicators
    
    def cleanup_expired_grants(self):
        """Remove expired capability grants."""
        expired = [
            grant_id for grant_id, grant in self.active_grants.items()
            if not grant.is_valid()
        ]
        for grant_id in expired:
            del self.active_grants[grant_id]

