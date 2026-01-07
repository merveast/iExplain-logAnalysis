
def inject_system_context(base_prompt, system_type):
    """
    Inject system-specific context into anomaly detector prompt.
    
    Args:
        base_prompt: Base anomaly detector prompt
        system_type: System type ('hdfs', 'kubernetes', etc.)
    
    Returns:
        Modified prompt with system context
    """
    SYSTEM_CONTEXT_TEMPLATES = {
                "hdfs": """
        System Context (HDFS Distributed Storage):
        Normal workflow: block allocation -> reception at datanodes -> writing -> replication (typically 3 copies) -> storage confirmation -> completion
        Common issues: I/O errors, incomplete replication, premature termination, missing storage confirmations
        """,
                "kubernetes": """
        System Context (Kubernetes Container Orchestration):
        Normal workflow: pod creation -> scheduling -> image pull -> container startup -> running state -> completion
        Common issues: ImagePullBackOff, CrashLoopBackOff, OOMKilled, scheduling failures, liveness probe failures
        """,
                "apache": """
        System Context (Apache Web Server):
        Normal workflow: request received -> processing -> resource access -> response generation -> connection closed
        Common issues: 4xx/5xx status codes, timeouts, connection resets, resource exhaustion, permission errors
        """,
                "database": """
        System Context (Database System):
        Normal workflow: query received -> parsing -> execution plan -> execution -> result return -> transaction commit
        Common issues: deadlocks, constraint violations, connection timeouts, replication lag, lock timeouts
        """,
    }
    
    if system_type and system_type.lower() in SYSTEM_CONTEXT_TEMPLATES:
        context = SYSTEM_CONTEXT_TEMPLATES[system_type.lower()]
        # Insert context before "Examples:" section
        if "Examples:" in base_prompt:
            parts = base_prompt.split("Examples:")
            return parts[0] + context + "\nExamples:" + parts[1]
    
    return base_prompt


def build_meta_explainer_prompt(block_id, raw_logs, parsed_logs, anomaly_output, explanation_output, system_type=None):
    """
    Build the prompt for meta-explainer agent.
    
    Args:
        block_id: Session identifier
        raw_logs: Original log content
        parsed_logs: Parsed log messages
        anomaly_output: Anomaly detector output (JSON)
        explanation_output: Event-level explanation
        system_type: Optional system type for context
    
    Returns:
        Formatted prompt string
    """
    intent = "Ensure system operations are healthy and explain any anomalies that violate reliability expectations."
    
    if system_type:
        intent = f"Ensure {system_type.upper()} operations are healthy and explain any anomalies that violate reliability expectations."
    
    prompt = f"""
        Generate a pipeline-level explanation for the following log analysis:

        Session ID: {block_id}

        Intent: {intent}

        Input Data:
        - Raw log entries: {len(raw_logs.split(chr(10)))} lines
        - Parsed log messages: {len(parsed_logs.split(chr(10)))} events

        Stage Outputs:

        Stage 1 - Log Parsing:
        {parsed_logs[:500]}...
        [truncated for brevity]

        Stage 2 - Anomaly Detection:
        {anomaly_output}

        Stage 3 - Event Explanation:
        {explanation_output}

        Provide a comprehensive pipeline-level explanation following the format specified in your system prompt.
        """
    return prompt
