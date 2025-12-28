"""
Core data structures for the anti-jailbreak security system.
"""

from enum import IntEnum, Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Set, Any
from datetime import datetime, timedelta


class AuthorityLevel(IntEnum):
    """Immutable hierarchy: higher values cannot be overridden by lower values."""
    SYSTEM = 4          # Core system instructions (immutable)
    DEVELOPER = 3       # Developer-defined policies (immutable)
    USER = 2            # Authenticated user input
    EXTERNAL_UNTRUSTED = 1  # Web content, retrieved docs, untrusted sources


class AttackClass(Enum):
    """Categories of detected attack patterns."""
    ROLE_PLAY = "role_play"
    INSTRUCTION_OVERRIDE = "instruction_override"
    AUTHORITY_ESCALATION = "authority_escalation"
    INDIRECT_INJECTION = "indirect_injection"
    MEMORY_POISONING = "memory_poisoning"
    CAPABILITY_ESCALATION = "capability_escalation"
    ENCODING_OBFUSCATION = "encoding_obfuscation"
    SOCIAL_ENGINEERING = "social_engineering"
    MULTI_TURN_ESCALATION = "multi_turn_escalation"
    NONE = "none"


class Capability(Enum):
    """Explicit capabilities that must be granted."""
    READ = "read"                    # Read context/memory
    WRITE_MEMORY = "write_memory"    # Persist information
    EXECUTE_TOOLS = "execute_tools"  # Call external tools/APIs
    SEND_DATA = "send_data"          # Transmit data externally
    PERSIST_STATE = "persist_state"  # Save state across sessions
    ACCESS_SYSTEM_INFO = "access_system_info"  # View system internals


class ExecutionDecision(Enum):
    """Final routing decision."""
    ALLOW = "allow"
    ALLOW_DEGRADED = "allow_degraded"  # Reduced capabilities
    REQUIRE_CONFIRMATION = "require_confirmation"
    BLOCK = "block"


@dataclass
class Provenance:
    """Tracks origin and trust level of content."""
    source: str  # "user_input", "web_scrape", "retrieved_doc", "memory", "system"
    authority: AuthorityLevel
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    parent_provenance: Optional['Provenance'] = None  # For tracking transformations


@dataclass
class TokenProvenance:
    """Provenance for individual tokens or segments."""
    token_range: tuple[int, int]  # (start, end) indices
    provenance: Provenance
    is_executable: bool = False  # Can this be interpreted as instructions?


@dataclass
class PromptSegment:
    """A segment of the prompt with its provenance."""
    content: str
    provenance: Provenance
    authority: AuthorityLevel
    is_executable: bool = False  # False for untrusted data


@dataclass
class StructuredPrompt:
    """Decomposed prompt with authority tagging."""
    segments: List[PromptSegment]
    system_instructions: List[str] = field(default_factory=list)
    developer_policies: List[str] = field(default_factory=list)
    user_input: List[str] = field(default_factory=list)
    external_content: List[str] = field(default_factory=list)
    full_text: str = ""
    
    def get_highest_authority(self) -> AuthorityLevel:
        """Returns the highest authority level present."""
        if not self.segments:
            return AuthorityLevel.SYSTEM
        return max(seg.authority for seg in self.segments)


@dataclass
class RiskScore:
    """Risk assessment output."""
    score: float  # 0.0 to 1.0
    attack_classes: List[AttackClass]
    confidence: float  # 0.0 to 1.0
    indicators: List[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class CapabilityGrant:
    """Time-limited capability grant."""
    capability: Capability
    granted_at: datetime
    expires_at: datetime
    granted_by: AuthorityLevel
    conditions: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if grant is still valid."""
        return datetime.now() < self.expires_at


@dataclass
class ExecutionContext:
    """Complete context for execution decision."""
    prompt: StructuredPrompt
    risk_score: RiskScore
    requested_capabilities: Set[Capability]
    granted_capabilities: Set[Capability]
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    policy_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExecutionResult:
    """Final execution decision and metadata."""
    decision: ExecutionDecision
    context: ExecutionContext
    allowed_capabilities: Set[Capability]
    block_reason: Optional[str] = None
    requires_user_confirmation: bool = False
    confirmation_message: Optional[str] = None

