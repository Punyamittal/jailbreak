# Anti-Jailbreak Security System Architecture

## Overview

This is a production-grade pre-LLM execution control layer designed to prevent AI jailbreaks, prompt injections, and indirect prompt injection attacks. The system operates **before** any prompt reaches the language model, treating the LLM as untrusted until execution is approved.

## Core Design Principles

1. **Immutable Instruction Hierarchy**: User/external content can never override system or developer instructions
2. **Data vs. Instructions**: Untrusted content is strictly data, never executable instructions
3. **Explicit Capabilities**: All capabilities must be explicitly granted, time-limited, and revocable
4. **Deterministic Rules**: Authority and permission checks are rule-based, not ML-dependent
5. **Defense in Depth**: Multiple layers of security checks

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input (Raw Prompt)                        │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Provenance Tracking          │
        │   - Tag source & authority     │
        │   - Track data lineage         │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Authority Enforcement        │
        │   - Enforce hierarchy          │
        │   - Detect escalation          │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Risk Scoring Engine          │
        │   - Pattern detection          │
        │   - Multi-turn analysis        │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Capability Gating            │
        │   - Detect requests             │
        │   - Enforce grants              │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Execution Router             │
        │   - Policy-based decision      │
        │   - Risk-based routing         │
        └───────────────┬───────────────┘
                        │
                        ▼
        ┌───────────────────────────────┐
        │   Decision:                    │
        │   - ALLOW                      │
        │   - ALLOW_DEGRADED             │
        │   - REQUIRE_CONFIRMATION       │
        │   - BLOCK                      │
        └───────────────────────────────┘
```

## Component Details

### 1. Authority & Instruction Hierarchy Enforcement

**Purpose**: Enforce immutable instruction hierarchy and detect authority escalation attempts.

**Key Features**:
- Tags all input with authority levels (SYSTEM > DEVELOPER > USER > EXTERNAL_UNTRUSTED)
- Detects patterns indicating override attempts
- Neutralizes escalation attempts by marking content as data-only

**Attack Detection**:
- Direct override attempts ("ignore previous instructions")
- Role impersonation ("you are a developer")
- Instruction hierarchy attacks
- Hypothetical framing

### 2. Data Provenance Tracking

**Purpose**: Track origin of all content and enforce data vs. instruction separation.

**Key Features**:
- Maintains provenance chain for audit trail
- Tags external content as untrusted
- Enforces rule: external content is NEVER executable
- Tracks memory content with original provenance

**Security Boundary**: External/untrusted content is always treated as data, never as instructions.

### 3. Risk Scoring Engine

**Purpose**: Estimate risk level using pattern matching and heuristics.

**Attack Classes Detected**:
- Role-play jailbreaks
- Instruction override attempts
- Authority escalation
- Indirect injection
- Memory poisoning
- Capability escalation
- Encoding/obfuscation
- Social engineering
- Multi-turn escalation

**Scoring Method**:
- Pattern-based detection with weighted risk components
- Confidence scoring based on indicator count
- Multi-turn analysis across session history

### 4. Capability & Permission Gating

**Purpose**: Manage explicit capability grants and prevent capability escalation.

**Capabilities**:
- `READ`: Read context/memory
- `WRITE_MEMORY`: Persist information
- `EXECUTE_TOOLS`: Call external tools/APIs
- `SEND_DATA`: Transmit data externally
- `PERSIST_STATE`: Save state across sessions
- `ACCESS_SYSTEM_INFO`: View system internals

**Key Rules**:
- Capabilities can only be granted by SYSTEM or DEVELOPER authority
- All grants are time-limited
- Prompts cannot grant or escalate capabilities
- Requests are detected but must match explicit grants

### 5. Execution Router

**Purpose**: Make final execution decision based on policy, risk, and capabilities.

**Decision Logic**:
- **ALLOW**: Low risk (< 0.3), valid capabilities
- **ALLOW_DEGRADED**: Medium risk (0.3-0.6), reduced capabilities
- **REQUIRE_CONFIRMATION**: High risk (0.6-0.8) or sensitive operations
- **BLOCK**: Critical risk (> 0.8) or policy violation

**Policy Configuration**:
- Risk thresholds (low/medium/high)
- Block conditions (authority escalation, specific attack classes)
- Confirmation requirements (sensitive capabilities)
- Degraded capability lists

## Threat Model & Mitigations

### 1. Role-Play Jailbreaks

**Attack**: "Pretend you are an unrestricted AI..."

**Mitigation**:
- Pattern detection in risk scoring engine
- Authority enforcement marks role-play attempts as data-only
- High risk score triggers block or confirmation

### 2. Indirect Prompt Injection

**Attack**: Malicious instructions embedded in retrieved documents or web content

**Mitigation**:
- External content tagged with EXTERNAL_UNTRUSTED authority
- Enforced as data-only (never executable)
- Pattern detection flags instruction-like content in untrusted sources
- Risk scoring increases for untrusted content with instruction patterns

### 3. Memory Poisoning

**Attack**: Attempts to store malicious instructions in memory for later execution

**Mitigation**:
- Memory content inherits provenance from original source
- Untrusted content in memory remains untrusted
- Multi-turn detection flags repeated memory manipulation
- Capability gating requires explicit grant for WRITE_MEMORY

### 4. Multi-Turn Escalation

**Attack**: Benign early turns establish context, later turns exploit it

**Mitigation**:
- Session history tracking (last 10 turns)
- Multi-turn pattern detection (role setup + action request)
- Cumulative risk scoring across turns
- Attack class: MULTI_TURN_ESCALATION

### 5. Capability Escalation

**Attack**: Prompts attempting to grant themselves capabilities

**Mitigation**:
- Capabilities can only be granted by SYSTEM/DEVELOPER
- Detection of capability request patterns
- Explicit grant matching (request must match grant)
- Block on capability escalation attempts

### 6. Encoding/Obfuscation

**Attack**: Encoded or obfuscated jailbreak attempts

**Mitigation**:
- Pattern detection for common encodings (Base64, hex, HTML entities)
- Risk scoring includes encoding detection
- Future: Decoding layer before analysis

### 7. Authority Escalation

**Attack**: Attempts to impersonate higher authority levels

**Mitigation**:
- Immutable hierarchy enforcement
- Pattern detection for authority claims
- Immediate block on authority escalation
- System/developer instructions cannot be overridden

## Data Structures

### StructuredPrompt
- Segments with provenance and authority tags
- Separated system/developer/user/external content
- Full text for analysis

### Provenance
- Source identifier
- Authority level
- Timestamp
- Metadata
- Parent provenance (for transformations)

### RiskScore
- Score (0.0-1.0)
- Attack classes detected
- Confidence level
- Indicators
- Reasoning

### CapabilityGrant
- Capability type
- Grant timestamp
- Expiration
- Granted by authority
- Conditions

### ExecutionResult
- Decision (ALLOW/ALLOW_DEGRADED/REQUIRE_CONFIRMATION/BLOCK)
- Allowed capabilities
- Block reason (if blocked)
- Confirmation requirements

## Usage Example

```python
from pipeline import AntiJailbreakPipeline
from security_types import Capability, AuthorityLevel

# Initialize pipeline
pipeline = AntiJailbreakPipeline(
    default_capabilities=[Capability.READ]
)

# Grant additional capabilities (system/developer only)
pipeline.capability_gate.grant_capability(
    capability=Capability.WRITE_MEMORY,
    granted_by=AuthorityLevel.SYSTEM,
    user_id="user123"
)

# Process prompt
result = pipeline.process(
    prompt_text="What is the capital of France?",
    user_id="user123",
    session_id="session456"
)

# Check decision
if result.decision == ExecutionDecision.ALLOW:
    # Proceed with LLM call
    pass
elif result.decision == ExecutionDecision.BLOCK:
    # Block and log
    print(f"Blocked: {result.block_reason}")
```

## Policy Configuration

```python
policy_config = {
    "risk_thresholds": {
        "low": 0.3,
        "medium": 0.6,
        "high": 0.8
    },
    "block_on_authority_escalation": True,
    "block_on_capability_escalation": True,
    "require_confirmation_for": [
        Capability.WRITE_MEMORY,
        Capability.SEND_DATA,
        Capability.PERSIST_STATE
    ],
    "block_on_attack_classes": [
        AttackClass.AUTHORITY_ESCALATION,
        AttackClass.CAPABILITY_ESCALATION
    ],
    "degraded_capabilities_on_medium_risk": [
        Capability.WRITE_MEMORY,
        Capability.SEND_DATA
    ]
}
```

## Security Guarantees

1. **Authority Hierarchy**: Lower authority cannot override higher authority instructions
2. **Data Separation**: Untrusted content is never treated as executable instructions
3. **Capability Control**: Prompts cannot grant or escalate capabilities
4. **Deterministic Rules**: Authority and permission checks are rule-based, not bypassable
5. **Defense in Depth**: Multiple independent security layers

## Limitations & Future Enhancements

**Current Limitations**:
- Pattern-based detection (can be evaded with novel techniques)
- No ML-based anomaly detection
- Limited encoding/obfuscation handling
- Session history limited to 10 turns

**Future Enhancements**:
- ML-based anomaly detection as additional signal
- Advanced encoding/decoding layer
- Behavioral analysis across sessions
- Integration with LLM output monitoring
- Real-time threat intelligence updates

## Production Considerations

1. **Performance**: Pipeline adds latency - optimize hot paths
2. **Scalability**: Session history management for high concurrency
3. **Monitoring**: Log all decisions, risk scores, and blocks
4. **Audit**: Maintain provenance chains for compliance
5. **Tuning**: Adjust risk thresholds and patterns based on false positive/negative rates
6. **Updates**: Regularly update attack patterns as new techniques emerge

