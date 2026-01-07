#!/usr/bin/env python3
"""
Run log analysis experiment with multi-agent pipeline:
 - Parser agent: removes headers from raw log messages
 - Anomaly detector agent: classifies sessions as normal/anomalous
 - Explanation generator agent: explains the classification decision
 - (Optional) Meta-explainer agent: provides pipeline-level explanation

Collects outputs, saves results, and evaluates against ground truth.
"""

import os
import time
import json
import argparse
from datetime import datetime
import config
from log_utils import (
    read_log_sessions,
    save_log_analysis_results,
    normalize_log_analysis_result,
    get_log_analysis_gt,
)
from agent_utils import create_agent
from ollama_utils import start_ollama_server, stop_ollama_server, start_ollama_server_log
from prompt_utils import inject_system_context
from evaluation import evaluate_and_save_log_analysis


def run_multi_agent_inference_log_analysis(
    log_sessions, 
    llm_config, 
    sys_prompt_parser, 
    sys_prompt_anomaly_detector, 
    sys_prompt_explanation_generator,
    exp_name, 
    result_dir,
    enable_meta_explainer=False,
    system_type=None
): 
    """
    Run multi-agent log analysis pipeline.
    
    Args:
        log_sessions: List of session dictionaries with 'block_id' and 'content'
        llm_config: LLM configuration for agents
        sys_prompt_*: System prompts for each agent
        exp_name: Experiment name for logging
        result_dir: Directory to save results
        enable_meta_explainer: Whether to run the meta-explainer agent
        system_type: Optional system type hint (e.g., 'hdfs', 'kubernetes', None for general)
    
    Returns:
        Tuple of (parser_results, anomaly_detector_results, explanation_generator_results, meta_explainer_results)
    """
    parser_results = []
    anomaly_detector_results = []
    explanation_generator_results = []
    meta_explainer_results = []

    if system_type:
        sys_prompt_anomaly_detector = inject_system_context(
            sys_prompt_anomaly_detector, 
            system_type
        )

    try:
        # Create agents
        print("[INFO] Creating agents...")
        log_parser = create_agent(
            agent_type="assistant",
            name="log_parser_agent",
            llm_config=llm_config,
            sys_prompt=sys_prompt_parser,
            #description=task_prompt,
        )
        anomaly_detector = create_agent(
            agent_type="assistant",
            name="log_anomaly_detector_agent",
            llm_config=llm_config,
            sys_prompt=sys_prompt_anomaly_detector,
        )
        explanation_generator = create_agent(
            agent_type="assistant",
            name="log_explanation_generator_agent",
            llm_config=llm_config,
            sys_prompt=sys_prompt_explanation_generator,
        )
        if enable_meta_explainer:
            meta_explainer = create_agent(
                agent_type="assistant",
                name="pipeline_meta_explainer_agent",
                llm_config=llm_config,
                sys_prompt="Generate a comprehensive explanation of how the entire log analysis pipeline processed a session and arrived at its conclusion, making the intent-to-outcome flow transparent.",
            )
        
        print(f"[INFO] Processing {len(log_sessions)} sessions...\n")
        # Process each session
        for i, session in enumerate(log_sessions):
            blk_id = session.get("block_id")
            log_content = session.get("content")
            print(f"\n--- Processing {blk_id} ({i+1}/{len(log_sessions)}) ---")
            # Stage 1: Parse logs
            parser_prompt = (
                "Extract only the message bodies by removing automatically generated headers "
                "(timestamp, log level, class, etc.) from the following log messages:\n\n"
                f"{log_content}"
            )
            res_parser = log_parser.generate_reply(messages=[{"content": parser_prompt, "role": "user"}])
            if res_parser is not None and "content" in res_parser:
                raw_output_parser = res_parser["content"].strip()
            else:
                raw_output_parser = "NONE"
                print(f"[WARNING] Parsing failed for {blk_id} - no response or invalid format.")

            parser_results.append({
                "block_id": blk_id,
                "raw_output": raw_output_parser,
            })

            # Stage 2: Detect anomalies
            anomaly_prompt = (
                "Analyze the following parsed log session and determine whether it represents "
                "NORMAL (0) or ANOMALOUS (1) behavior:\n\n"
                f"Parsed Session Logs:\n{raw_output_parser}"
            )
            
            res_anomaly_detector = anomaly_detector.generate_reply(
                messages=[{"content": anomaly_prompt, "role": "user"}]
            )
            if res_anomaly_detector is not None and "content" in res_anomaly_detector:
                raw_output_anomaly = res_anomaly_detector["content"].strip()
            else:
                raw_output_anomaly = '{"label": -1, "signals": ["detection_failed"]}'
                print(f"[WARNING] Anomaly detection failed for {blk_id}")
            
            anomaly_detector_results.append({
                "block_id": blk_id,
                "raw_output": raw_output_anomaly,
            })

            # Stage 3: Generate explanation
            explanation_prompt = (
                "Explain why the following log session was classified as NORMAL or ANOMALOUS:\n\n"
                f"Parsed Session Logs:\n{raw_output_parser}\n\n"
                f"Anomaly Detection Output:\n{raw_output_anomaly}"
            )
            
            res_explanation_generator = explanation_generator.generate_reply(
                messages=[{"content": explanation_prompt, "role": "user"}]
            )
            if res_explanation_generator is not None and "content" in res_explanation_generator:
                raw_output_explanation = res_explanation_generator["content"].strip()
            else:
                raw_output_explanation = "EXPLANATION_FAILED"
                print(f"[WARNING] Explanation generation failed for {blk_id}")

            explanation_generator_results.append({
                "block_id": blk_id,
                "raw_output": raw_output_explanation,
            })
            # Optional Stage 4: Meta-explainer
            # TODO: Implement if needed

            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"[PROGRESS] Completed {i + 1}/{len(log_sessions)} sessions")
    except Exception as e:
        print(f"[ERROR] Pipeline execution failed: {e}")
        raise

    finally:
        return parser_results, anomaly_detector_results, explanation_generator_results

def parse_args():
    p = argparse.ArgumentParser(description="Multi-agent log analysis runner")
    p.add_argument("--input", default="HDFS_385_sampled_sessions", 
                   help="Input sessions directory name (in config.DATA_DIR)")
    p.add_argument("--gt", default="HDFS_anomaly_label_385_session_sampled.csv", 
                   help="Ground-truth file name (in config.DATA_DIR)")
    p.add_argument("--result-dir", default=config.RESULT_DIR, 
                   help="Directory to store results")
    p.add_argument("--design", default=None, 
                   help="Experiment design name. If omitted, MA-{shot} will be used.")
    p.add_argument("--shot", default="few", choices=["zero", "few"], 
                   help="Zero-shot or few-shot prompt selection")
    p.add_argument("--system-type", default=None, 
                   choices=["hdfs", "kubernetes", "apache", "database", None],
                   help="System type for context injection (optional)")
    p.add_argument("--enable-meta-explainer", action="store_true",
                   help="Enable pipeline meta-explainer agent")
    return p.parse_args()


def main():
    args = parse_args()
    TASK = "log-analysis"

    shot = args.shot.lower()
    if shot not in ("zero", "few"):
        raise SystemExit("--shot must be 'zero' or 'few'")

    # Select prompts based on shot
    if shot == "zero":
        sys_prompt_parser = config.SYS_MSG_LOG_PREPROCESSOR_ZERO_SHOT
        sys_prompt_anomaly_detector = config.SYS_MSG_LOG_ANOMALY_DETECTOR_SIMPLIFIED_ZERO_SHOT
        sys_prompt_explanation_generator = config.SYS_MSG_LOG_EXPLANATION_GENERATOR_ZERO_SHOT
    else:
        sys_prompt_parser = config.SYS_MSG_LOG_PREPROCESSOR_FEW_SHOT
        sys_prompt_anomaly_detector = config.SYS_MSG_LOG_ANOMALY_DETECTOR_SIMPLIFIED_FEW_SHOT
        sys_prompt_explanation_generator = config.SYS_MSG_LOG_EXPLANATION_GENERATOR_FEW_SHOT
    
    # Meta-explainer prompt (if enabled)
    sys_prompt_meta_explainer = getattr(config, 'SYS_MSG_PIPELINE_META_EXPLAINER_FEW_SHOT', None)

    llm_config = config.LLM_CONFIG
    DATA_DIR = config.DATA_DIR
    RESULT_DIR = args.result_dir
    os.makedirs(RESULT_DIR, exist_ok=True)

    design = args.design
    if design is None:
        design = f"MA-{shot}"
        if args.system_type:
            design += f"-{args.system_type}"
        if args.enable_meta_explainer:
            design += "-meta"

    project_name = f"{TASK}_{design}"
    model_name = llm_config["config_list"][0]["model"]
    model = model_name.replace(":", "-")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    exp_name = f"{project_name}_{model}_{timestamp}"

    input_dir_path = os.path.join(DATA_DIR, args.input)
    ground_truth_file_path = os.path.join(DATA_DIR, args.gt)

    # Read inputs
    print(f"[INFO] Loading log sessions from {input_dir_path}...")
    log_sessions = read_log_sessions(input_dir_path)
    print(f"[INFO] Loaded {len(log_sessions)} sessions")

    # Start Ollama server
    print("[INFO] Starting Ollama server...")
    #proc = start_ollama_server()
    #time.sleep(5)
    
    try:
        # Run multi-agent inference
        print(f"[INFO] Starting multi-agent pipeline (experiment: {exp_name})...")
        results = run_multi_agent_inference_log_analysis(
            log_sessions=log_sessions,
            llm_config=llm_config,
            sys_prompt_parser=sys_prompt_parser,
            sys_prompt_anomaly_detector=sys_prompt_anomaly_detector,
            sys_prompt_explanation_generator=sys_prompt_explanation_generator,
            exp_name=exp_name,
            result_dir=RESULT_DIR,
            enable_meta_explainer=args.enable_meta_explainer,
            system_type=args.system_type
        )
        
        parser_results, anomaly_results, explanation_results, meta_results = results
        
        print(f"\n[INFO] Pipeline completed successfully!")
        print(f"  - Parsed: {len(parser_results)} sessions")
        print(f"  - Detected: {len(anomaly_results)} sessions")
        print(f"  - Explained: {len(explanation_results)} sessions")
        if meta_results:
            print(f"  - Meta-explained: {len(meta_results)} sessions")
        
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        raise
    finally:
        print("[INFO] Stopping Ollama server...")
        #stop_ollama_server(proc)
    
    # Save and normalize predictions
    print(f"[INFO] Saving results to {RESULT_DIR}...")
    log_anomaly_results = (parser_results, anomaly_results, explanation_results)
    
    normalized_results = save_log_analysis_results(
        log_anomaly_results,
        normalize_log_analysis_result,
        exp_name,
        llm_config,
        out_dir=RESULT_DIR,
    )
    
    # Save meta-explainer results separately if enabled
    if meta_results and args.enable_meta_explainer:
        meta_output_path = os.path.join(RESULT_DIR, f"{exp_name}_meta_explanations.json")
        with open(meta_output_path, 'w') as f:
            json.dump(meta_results, f, indent=2)
        print(f"[INFO] Meta-explanations saved to {meta_output_path}")

    # Load ground truth and evaluate
    print(f"[INFO] Loading ground truth from {ground_truth_file_path}...")
    gt = get_log_analysis_gt(ground_truth_file_path)
    
    print("[INFO] Evaluating predictions...")
    evaluate_and_save_log_analysis(gt, normalized_results, exp_name, RESULT_DIR)
    
    print(f"\n[SUCCESS] Experiment completed: {exp_name}")
    print(f"[SUCCESS] Results saved in: {RESULT_DIR}")


if __name__ == "__main__":
    main()