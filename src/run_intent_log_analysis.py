#!/usr/bin/env python3
"""
Run two-agent intent-based log analysis experiment:
 - Intent-Aware Log Analyzer: correlates JSON log events against a user intent and its timestamp,
   classifying fulfillment status and extracting signals.
 - Explanation Generator: produces a human-readable explanation of the fulfillment outcome.

Intent input:
    For now, INTENT and INTENT_TS are defined as variables below.
    Later, these will be received from the user or an external system.
"""

import os
import json
import time
import argparse
from datetime import datetime
import config
from log_utils import (
    read_jsonl_log_file,
    save_intent_analysis_results,
)
from agent_utils import create_agent
from ollama_utils import start_ollama_server_log, stop_ollama_server


# ---------------------------------------------------------------------------
# Intent configuration
# TODO: replace with dynamic input from user / external system
# ---------------------------------------------------------------------------

INTENT    = "What happened on the machines following my previous intent?"
INTENT_TS = "2026-02-23T11:32:00+00:00"


# ---------------------------------------------------------------------------
# Core two-agent pipeline
# ---------------------------------------------------------------------------

def run_intent_log_analysis(
    session_id,
    intent,
    intent_ts,
    log_events,
    llm_config,
    sys_prompt_analyzer,
    sys_prompt_explainer,
):
    """
    Runs the two-agent pipeline for a single log session.

    Agent 1 — Intent-Aware Log Analyzer:
        Receives log events + intent context, returns fulfillment JSON.
    Agent 2 — Explanation Generator:
        Receives log events + intent context + analyzer output, returns
        a structured natural-language explanation.
    """
    analyzer_agent = create_agent(
        agent_type="assistant",
        name="intent_log_analyzer_agent",
        llm_config=llm_config,
        sys_prompt=sys_prompt_analyzer,
    )
    explainer_agent = create_agent(
        agent_type="assistant",
        name="intent_explanation_agent",
        llm_config=llm_config,
        sys_prompt=sys_prompt_explainer,
    )

    print(f"\n--- Processing session: {session_id} ---")
    print(f"    Intent           : {intent}")
    print(f"    Intent timestamp : {intent_ts}")
    print(f"    Log events       : {len(log_events)}")

    # ------------------------------------------------------------------
    # Agent 1 — Intent-Aware Log Analyzer
    # ------------------------------------------------------------------
    analyzer_prompt = _build_analyzer_prompt(intent, intent_ts, log_events)
    res_analyzer = analyzer_agent.generate_reply(
        messages=[{"role": "user", "content": analyzer_prompt}]
    )

    if res_analyzer and "content" in res_analyzer:
        raw_analyzer_output = res_analyzer["content"].strip()
    else:
        raw_analyzer_output = json.dumps({
            "fulfillment_status": "UNKNOWN",
            "pre_intent_context": [],
            "post_intent_events": [],
            "intent_related_events": [],
            "unrelated_or_unexpected_events": [],
            "signals": ["No response from analyzer agent"],
        })
        print(f"[Warning] Analyzer returned no response for session '{session_id}'.")

    print(f"Analyzer Output:\n{raw_analyzer_output}")

    # ------------------------------------------------------------------
    # Agent 2 — Explanation Generator
    # ------------------------------------------------------------------
    explainer_prompt = _build_explainer_prompt(intent, intent_ts, log_events, raw_analyzer_output)
    res_explainer = explainer_agent.generate_reply(
        messages=[{"role": "user", "content": explainer_prompt}]
    )

    if res_explainer and "content" in res_explainer:
        raw_explainer_output = res_explainer["content"].strip()
    else:
        raw_explainer_output = "No explanation generated."
        print(f"[Warning] Explainer returned no response for session '{session_id}'.")

    print(f"Explanation Output:\n{raw_explainer_output}")

    return {
        "session_id":         session_id,
        "intent":             intent,
        "intent_ts":          intent_ts,
        "analyzer_output":    raw_analyzer_output,
        "explanation_output": raw_explainer_output,
    }


# ---------------------------------------------------------------------------
# Prompt builders
# ---------------------------------------------------------------------------

def _build_analyzer_prompt(intent: str, intent_ts: str, log_events: list) -> str:
    events_str = "\n".join(json.dumps(e) for e in log_events)
    return (
        f"User Intent: {intent}\n"
        f"Intent Timestamp: {intent_ts}\n\n"
        f"Log Events (ordered by timestamp):\n{events_str}\n\n"
        "Analyze the log events relative to the intent and its timestamp. "
        "Separate events into PRE-INTENT and POST-INTENT groups, identify which "
        "post-intent events are directly related to the intent, flag any unrelated "
        "or unexpected changes, and return your analysis as JSON."
    )


def _build_explainer_prompt(
    intent: str,
    intent_ts: str,
    log_events: list,
    analyzer_output: str,
) -> str:
    events_str = "\n".join(json.dumps(e) for e in log_events)
    return (
        f"User Intent: {intent}\n"
        f"Intent Timestamp: {intent_ts}\n\n"
        f"Log Events (ordered by timestamp):\n{events_str}\n\n"
        f"Analyzer Output:\n{analyzer_output}\n\n"
        "Using the above information, produce a structured explanation of the "
        "fulfillment outcome following the format defined in your instructions."
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Two-agent intent-based node log analysis runner")
    p.add_argument(
        "--log-file",
        required=True,
        help="Path to the raw JSONL log file (e.g. /path/to/2026-02-23.log)",
    )
    p.add_argument(
        "--result-dir",
        default=config.RESULT_DIR,
        help="Directory to store results",
    )
    p.add_argument(
        "--design",
        default=None,
        help="Experiment design name. Defaults to DA-{shot}.",
    )
    p.add_argument(
        "--shot",
        default="zero",
        choices=["zero", "few"],
        help="Prompt variant: zero-shot or few-shot",
    )
    return p.parse_args()


def main():
    args = parse_args()
    shot = args.shot.lower()

    if shot == "zero":
        sys_prompt_analyzer  = config.SYS_MSG_INTENT_LOG_ANALYZER_ZERO_SHOT
        sys_prompt_explainer = config.SYS_MSG_INTENT_EXPLANATION_GENERATOR_ZERO_SHOT
    else:
        sys_prompt_analyzer  = config.SYS_MSG_INTENT_LOG_ANALYZER_FEW_SHOT
        sys_prompt_explainer = config.SYS_MSG_INTENT_EXPLANATION_GENERATOR_FEW_SHOT

    llm_config = config.LLM_CONFIG
    RESULT_DIR = args.result_dir
    os.makedirs(RESULT_DIR, exist_ok=True)

    design     = args.design if args.design else f"DA-{shot}"
    model_name = llm_config["config_list"][0]["model"]
    model      = model_name.replace(":", "-")
    timestamp  = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name   = f"intent-log-analysis_{design}_{model}_{timestamp}"

    # Derive a session_id from the log filename (e.g. "2026-02-23")
    session_id = os.path.splitext(os.path.basename(args.log_file))[0]

    # Read raw log events from JSONL file
    log_events = read_jsonl_log_file(args.log_file)
    print(f"Loaded {len(log_events)} log event(s) from '{args.log_file}'")

    proc = start_ollama_server_log()
    time.sleep(5)

    try:
        result = run_intent_log_analysis(
            session_id=session_id,
            intent=INTENT,
            intent_ts=INTENT_TS,
            log_events=log_events,
            llm_config=llm_config,
            sys_prompt_analyzer=sys_prompt_analyzer,
            sys_prompt_explainer=sys_prompt_explainer,
        )
    except Exception as e:
        print(f"[Error] Inference failed: {e}")
        stop_ollama_server(proc)
        raise

    save_intent_analysis_results([result], exp_name, llm_config, out_dir=RESULT_DIR)

    stop_ollama_server(proc)
    print(f"\nExperiment '{exp_name}' complete. Results saved to: {RESULT_DIR}")


if __name__ == "__main__":
    main()