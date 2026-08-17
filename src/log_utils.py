import os
import csv
import re
import json
from datetime import datetime
from collections import Counter
from typing import List, Optional, Dict, Any, Tuple


def read_log_messages(file_path: str) -> List[str]:
    """Read log messages from a file, one per line."""
    log_messages = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            log_messages = [line.strip() for line in file]
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return log_messages


def slice_log_file(input_path: str, num_lines: int) -> Optional[str]:
    """
    Slices the first `num_lines` from a log file and saves it to a new file.
    Returns the path to the new sliced file.
    """
    if not os.path.exists(input_path):
        print(f"Error: The file '{input_path}' does not exist.")
        return None
    try:
        with open(input_path, 'r', encoding='utf-8') as infile:
            lines = [next(infile).strip() for _ in range(num_lines)]
        dir_path, filename = os.path.split(input_path)
        name, ext = os.path.splitext(filename)
        new_filename = f"{name}_{num_lines}{ext}"
        new_path = os.path.join(dir_path, new_filename)
        with open(new_path, 'w', encoding='utf-8') as outfile:
            outfile.write('\n'.join(lines) + '\n')
        print(f"Sliced file created: {new_path}")
        return new_path
    except StopIteration:
        print(f"Warning: The file has fewer than {num_lines} lines. All lines were copied.")
    except Exception as e:
        print(f"An error occurred: {e}")
    return None

def _clean_text(text: str) -> str:
    """Helper to remove extra whitespace and newlines."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    cleaned = " ".join(lines)
    return re.sub(r"\s+", " ", cleaned).strip()

def normalize_template_old(text: str) -> str:
    """Normalize template output (legacy version)."""
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"You are a helpful assistant\.?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = re.split(
        r"The template you provided is correct|Therefore, no further suggestions",
        text
    )[0]
    return _clean_text(text)

def normalize_template(text: str) -> str:
    """Normalize template output (main version)."""
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"You are a helpful assistant\.?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    template_match = re.match(r'^([^H]*?<\*>.*?<\*>[^H]*?)(?=Human|$)', text, re.DOTALL | re.IGNORECASE)
    if template_match:
        text = template_match.group(1)
    else:
        text = re.sub(r"Human Compare and refine.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    patterns = [
        r"\*\*Final Refined Template:\*\*.*?(?:\n|$)",
        r"Final Refined Template:.*?(?:\n|$)",
        r"Both templates are correct.*?(?:\n|$)",
        r"Merged and corrected.*?(?:\n|$)",
        r"^(The template is (incorrect|correct)\. (The )?correct template should be:)\s*",
        r"^The template corresponding to the log message is:\s*",
        r"^The template corresponding to the log message would be:\s*",
        r"\s*Here, <\*> represents.*$",
        r'"\s*This means.*$',
        r"```python.*?```",
        r"^python\s+import\s+.*",
        r"(?i)I am an AI model.*?(?=\n|$)",
        r"\*\*Created Question\*\*:.*",
        r"\*\*Created Answer\*\*:.*"
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.IGNORECASE | re.DOTALL)
    quoted_match = re.match(r'^[\'"`](.+?)[\'"`]\.\s+This\b.*', text, flags=re.IGNORECASE | re.DOTALL)
    if quoted_match:
        return quoted_match.group(1).strip()
    text = re.split(
        r'\s+(Here,.*?|This means|This can be interpreted|In this case|The angle brackets)\b',
        text,
        maxsplit=1,
        flags=re.IGNORECASE | re.DOTALL
    )[0]
    text = re.sub(r"^['\"](.*?)['\"]$", r"\1", text.strip())
    text = re.sub(r'[.,;:\s]+$', '', text.strip())
    explanation_patterns = [
        r"This template better abstracts.*?parameters?\.",
        r"This template correctly abstracts.*?\.",
        r"This template is more accurate.*?\.",
        r"Both templates are correct.*?preferred\.",
        r"You are a language model.*",
        r"You are Qwen, created by Alibaba Cloud.*",
        r"\*\*Created Question\*\*:*",
        r"The template you provided is correct",
        r"Therefore, no further suggestions",
        r"This template is more accurate.*",
        r"(Merged and corrected to abstract both dynamic parameters)",
        r"(Merged and corrected both templates into a more accurate version)",
        r"This template better abstracts the log message by*",
        r"This template better abstracts the log message by including the dynamic port number in the IP address field\.",
        r"Both templates are correct, but the Parser Agent's template is more abstract as it does not include a specific path\. Therefore, the Parser Agent's template is chosen as the final refined version\.",
        r"Both templates are correct, but the Parser Agent's template is more abstract as it does not specify the path, making it applicable to any similar log message without modification\.", 
        r"Both templates are correct, but the Parser Agent's template is more abstract as it does not specify the path, making it applicable to any path\.",
        r"Both templates are correct, but the Parser Agent's template better abstracts the dynamic parameters\.",
        r"Both templates are correct and abstract the dynamic parameters effectively\. No merging is necessary as they already capture all the essential information from the log message\\."
    ]
    for pattern in explanation_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.DOTALL)

    # Keep only the part before any explanatory "Note:" or similar
    text = re.split(r"\bNote:|This template|Thus,|Therefore,|The template reflects", text, maxsplit=1, flags=re.IGNORECASE)[0]

    return _clean_text(text)

def normalize_template_v1(text: str) -> str:
    """Normalize template output (v1)."""
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"You are a helpful assistant\.?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    patterns = [
        (r"Here is an example of a log message and its corresponding template:.*?Template:\s*([^\n]+)", 1),
        (r"The template remains as it is:\s*([^\n]+)", 1),
        (r"([^\n]+)\s+This is the template corresponding to the log message", 1),
        (r'The template should be ["\']([^"\']+)["\']', 1),
        (r'The template should be\s+([^\n.]+?)(?:\.|$)', 1),
        (r"should indeed be as follows:\s*([^\n]+)", 1),
        (r'print\((?:template\s*=\s*)?[\'"]([^\'"]+)[\'"]\)', 1),
        (r'template\s*=\s*[\'"]([^\'"]+)[\'"]', 1)
    ]
    for pat, group in patterns:
        match = re.search(pat, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(group).strip()
    template_match = re.match(r'^([^H]*?<(?:\*|[^>]*)>.*?<(?:\*|[^>]*)>[^H]*?)(?=Human|$)', text, re.DOTALL | re.IGNORECASE)
    if template_match:
        text = template_match.group(1)
    else:
        text = re.sub(r"Human Compare and refine.*", "", text, flags=re.IGNORECASE | re.DOTALL)
    # ...repeat cleaning as above...
    return normalize_template(text)

def normalize_template_v2(text: str) -> str:
    """Normalize template output (v2)."""
    text = re.sub(r"<\|.*?\|>", "", text)
    text = re.sub(r"You are a helpful assistant\.?", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")
    text = re.sub(
        r"^(I apologize.*?|Yes, (that's|you are|you're) correct.*?|You're right.*?|Indeed,.*?)(?=\s|$)",
        "",
        text,
        flags=re.IGNORECASE | re.DOTALL
    )
    patterns = [
        r"Here is an example of a log message and its corresponding template:.*?Template:\s*([^\n]+)",
        r"The template remains as it is:\s*([^\n]+)",
        r"([^\n]+)\s+This is the template corresponding to the log message",
        r'The template should be ["\']([^"\']+)["\']',
        r'The template should be\s+([^\n.]+?)(?:\.|$)',
        r"should indeed be as follows:\s*([^\n]+)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    inline_backtick_match = re.search(r"`([^`]*<[^`>]*>[^`]*)`", text)
    if inline_backtick_match:
        return inline_backtick_match.group(1).strip()
    quoted_template_match = re.search(r'"([^"]*<[^>]+>[^"]*)"', text)
    if quoted_template_match:
        return quoted_template_match.group(1).strip()
    single_quoted_match = re.search(r"'([^']*<[^>]+>[^']*)'", text)
    if single_quoted_match:
        return single_quoted_match.group(1).strip()
    # ...repeat cleaning as above...
    return normalize_template(text)

def generate_filenames(design: str, llm_model: str, base_dir: str = ".", suffixes: Tuple[str, ...] = ("raw", "normalized")) -> Dict[str, str]:
    """Generate filenames for raw and normalized template outputs."""
    model_name = llm_model.replace(":", "_").replace("/", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return {
        suffix: os.path.join(base_dir, f"{design}_{model_name}_{timestamp}_{suffix}.txt")
        for suffix in suffixes
    }

def save_templates(parsed_templates: List[str], llm_config: Dict[str, Any], design: str, output_dir: str = "templates_output") -> Tuple[str, str]:
    """Save raw and normalized templates to files."""
    os.makedirs(output_dir, exist_ok=True)
    model = llm_config["config_list"][0]["model"].replace(":", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_filename = f"{output_dir}/{design}_{model}_{timestamp}_raw.txt"
    normalized_filename = f"{output_dir}/{design}_{model}_{timestamp}_normalized.txt"
    normalized_templates = [normalize_template(t) for t in parsed_templates]
    with open(raw_filename, "w", encoding="utf-8") as f:
        for t in parsed_templates:
            f.write(t.strip() + "\n")
    with open(normalized_filename, "w", encoding="utf-8") as f:
        for t in normalized_templates:
            f.write(t + "\n")
    print(f"Saved {len(parsed_templates)} raw and {len(normalized_templates)} normalized templates.")
    return raw_filename, normalized_filename


def extract_last_template_from_history(history: List[Dict[str, Any]], agent_name: str = 'log_parser_agent') -> Optional[str]:
    """Extract last plausible template message from agent history."""
    TEMPLATE_PATTERN = re.compile(r'<\*>|blk_<\*>|<.*?>')
    for msg in reversed(history):
        if msg['name'] == agent_name and TEMPLATE_PATTERN.search(msg['content'].strip()):
            return msg['content'].strip()
    return None


def extract_last_template_from_history_loose(history: List[Dict[str, Any]], agent_name: str = 'log_parser_agent') -> Optional[str]:
    """Extract last non-empty message from agent history."""
    for msg in reversed(history):
        if msg['name'] == agent_name:
            content = msg['content'].strip()
            if content:
                return content
    return None

def extract_template_from_parser_responses(parser_responses: List[str]) -> str:
    """Extract last valid template from parser responses."""
    for response in reversed(parser_responses):
        content = response.strip()
        if '<*>' in content and not re.search(r'understood|no further feedback|thank|feel free|additional feedback', content, re.IGNORECASE):
            content = re.sub(r"^```|```$", "", content.strip(), flags=re.MULTILINE)
            return content
    return "NONE"

def extract_event_templates(csv_file_path: str) -> List[str]:
    """Extracts a list of all EventTemplates from a CSV file."""
    event_templates = []
    with open(csv_file_path, mode='r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            event_template = row.get("EventTemplate")
            if event_template:
                event_templates.append(event_template.strip())
    return event_templates

def read_log_sessions(input_dir):
    """
    Reads all .log files under input_dir and returns a list of dicts with block_id and log content.
    Each .log file corresponds to one HDFS block session.
    """
    sessions = []
    for fname in sorted(os.listdir(input_dir)):
        if fname.endswith(".log"):
            block_id = fname.replace(".log", "")
            file_path = os.path.join(input_dir, fname)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            sessions.append({
                "block_id": block_id,
                "content": content
            })
    print(f"[INFO] Loaded {len(sessions)} log sessions from {input_dir}")
    return sessions

def save_parsed_sessions(sessions, out_dir, exp_name):
    """
    Saves parsed log sessions to a JSON file for reproducibility.
    """
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{exp_name}_parsed_sessions.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(sessions, f, indent=2)
    print(f"[INFO] Saved parsed sessions to {out_path}")
    return out_path


def get_log_analysis_gt(gt_file_path):
    """
    Loads ground truth CSV file with BlockId, Label.
    Returns a dict {block_id: "0"/"1"}.
    """
    gt = {}
    with open(gt_file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row["Label"].strip().lower()
            gt[row["BlockId"]] = "1" if label == "anomaly" else "0"
    print(f"[INFO] Loaded ground truth for {len(gt)} block IDs from {gt_file_path}")
    return gt

def save_log_analysis_results(results, normalize_fn, exp_name, llm_config, out_dir="results"):
    """
    Saves raw and normalized anomaly detection results.
    """
    os.makedirs(out_dir, exist_ok=True)
    model = llm_config["config_list"][0]["model"].replace(":", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path = os.path.join(out_dir, f"{exp_name}_{model}_{timestamp}_raw.txt")
    normalized_path = os.path.join(out_dir, f"{exp_name}_{model}_{timestamp}_normalized.txt")

    normalized = []
    with open(raw_path, "w", encoding="utf-8") as fr, open(normalized_path, "w", encoding="utf-8") as fn:
        for item in results:
            block_id, raw_output = item["block_id"], item["raw_output"]
            print(f"[DEBUG] Block ID: {block_id}, Raw Output: {raw_output}")
            normalized_label = normalize_fn(raw_output)
            print(f"[DEBUG] Block ID: {block_id}, Normalized Label: {normalized_label}")
            normalized.append({"block_id": block_id, "normalized": normalized_label})
            fr.write(f"{block_id}\t{raw_output.strip()}\n")
            fn.write(f"{block_id}\t{normalized_label}\n")

    print(f"[INFO] Saved raw results to {raw_path}")
    print(f"[INFO] Saved normalized results to {normalized_path}")
    return normalized

def normalize_log_analysis_result(text):
    """
    Normalize LLM outputs to 0 or 1.
    Removes extra explanations and keeps only a valid binary digit.
    """
    if text is None:
        return "0"

    # Convert to string and trim
    text = str(text).strip()

    # Remove code block markers and other formatting
    text = re.sub(r"```[a-z]*|```", "", text)

    # Remove long numeric sequences (e.g., timestamps, block IDs)
    text = re.sub(r"\d{4,}", " ", text)

    # Replace all non-digit characters with spaces
    text = re.sub(r"[^\d]", " ", text)

    # Extract all standalone 0s and 1s and take the last one
    matches = re.findall(r"\b[01]\b", text)
    if matches:
        return matches[-1]

    # Fallback: interpret based on words
    if re.search(r"normal", text, re.I):
        return "0"
    if re.search(r"anomal", text, re.I):
        return "1"
    print("Could not determine label, defaulting to 0")
    return "0"  # Default to normal if uncertain


def normalize_log_analysis_result_json(raw_output):
    """
    Parse LLM JSON output and extract:
      - label: int (0 or 1)
      - signals: list[str]
    """
    if raw_output is None:
        return {
            "label": 1,
            "signals": ["missing model output"]
        }

    raw_output = str(raw_output).strip()

    # Remove markdown code fences if present
    raw_output = re.sub(r"```json|```", "", raw_output).strip()

    # Extract first JSON object from text
    match = re.search(r"\{.*\}", raw_output, re.DOTALL)

    print("RAW OUTPUT >>>")
    print(raw_output)
    print("<<< END RAW")
    if not match:
        return {
            "label": 0,
            "signals": ["invalid json output"]
        }

    json_str = match.group(0)

    try:
        parsed = json.loads(json_str)

        label = parsed.get("label")
        signals = parsed.get("signals", [])

        if label in (0, 1):
            return {
                "label": int(label),
                "signals": signals if isinstance(signals, list) else []
            }

        return {
            "label": 0,
            "signals": ["label not 0 or 1"]
        }

    except json.JSONDecodeError:
        return {
            "label": 0,
            "signals": ["invalid json output"]
        }

def save_log_analysis_results_json(results, normalize_fn, exp_name, llm_config, out_dir="results"):
    os.makedirs(out_dir, exist_ok=True)

    model = llm_config["config_list"][0]["model"].replace(":", "-")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    raw_path = os.path.join(out_dir, f"{exp_name}_{model}_{timestamp}_raw.txt")
    normalized_path = os.path.join(out_dir, f"{exp_name}_{model}_{timestamp}_normalized.json")

    normalized = []

    with open(raw_path, "w", encoding="utf-8") as fr:
        for item in results:
            block_id = item["block_id"]
            raw_output = item["raw_output"]

            parsed = normalize_fn(raw_output)

            entry = {
                "block_id": block_id,
                "normalized": parsed["label"],
                "signals": parsed["signals"]
            }
            #print(f"[DEBUG] Block ID: {block_id}, Raw Output: {raw_output}, Parsed: {entry}")
            print(f"[DEBUG] Block ID: {block_id}, Normalized Label: {entry['normalized']}, Signals: {entry['signals']}")
            normalized.append(entry)

            fr.write(f"{block_id}\t{raw_output}\n")

    # Save normalized + signals as JSON
    with open(normalized_path, "w", encoding="utf-8") as fn:
        json.dump(normalized, fn, indent=2)

    print(f"[INFO] Saved raw results to {raw_path}")
    print(f"[INFO] Saved normalized results to {normalized_path}")

    return normalized



def read_intent_log_sessions(input_dir: str) -> list[dict]:
    """
    Read all intent log sessions from a directory.

    Supports two file layouts:

    Layout A — unified JSON (preferred):
        Each .json file contains a single session object with keys:
            session_id, intent, intent_ts, log_events (list of event dicts)

    Layout B — paired files:
        Each session has a JSONL file of raw log events (<name>.jsonl)
        and a sidecar metadata file (<name>.meta.json) containing:
            session_id, intent, intent_ts

    Sessions are returned sorted by session_id for reproducible ordering.
    Malformed files are skipped with a warning.
    """
    sessions = []

    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    all_files = sorted(os.listdir(input_dir))

    # --- Layout A: unified .json files ---
    json_files = [f for f in all_files if f.endswith(".json") and not f.endswith(".meta.json")]
    for fname in json_files:
        fpath = os.path.join(input_dir, fname)
        try:
            with open(fpath, "r", encoding="utf-8") as fh:
                session = json.load(fh)

            # Validate required keys
            for key in ("intent", "intent_ts", "log_events"):
                if key not in session:
                    raise ValueError(f"Missing required key '{key}'")

            # Assign session_id from filename if not present
            if "session_id" not in session:
                session["session_id"] = os.path.splitext(fname)[0]

            sessions.append(session)

        except Exception as e:
            print(f"[Warning] Skipping {fname}: {e}")

    # --- Layout B: paired .jsonl + .meta.json files ---
    jsonl_files = [f for f in all_files if f.endswith(".jsonl")]
    for fname in jsonl_files:
        base      = fname[: -len(".jsonl")]
        meta_path = os.path.join(input_dir, f"{base}.meta.json")
        jsonl_path = os.path.join(input_dir, fname)

        # Skip if already loaded as Layout A
        if os.path.exists(os.path.join(input_dir, f"{base}.json")):
            continue

        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)

            log_events = []
            with open(jsonl_path, "r", encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        log_events.append(json.loads(line))

            sessions.append({
                "session_id": meta.get("session_id", base),
                "intent":     meta["intent"],
                "intent_ts":  meta["intent_ts"],
                "log_events": log_events,
            })

        except FileNotFoundError:
            print(f"[Warning] Skipping {fname}: no matching .meta.json found at {meta_path}")
        except Exception as e:
            print(f"[Warning] Skipping {fname}: {e}")

    sessions.sort(key=lambda s: s.get("session_id", ""))
    print(f"[read_intent_log_sessions] Loaded {len(sessions)} session(s) from '{input_dir}'")
    return sessions


def save_intent_analysis_results(
    results: list[dict],
    exp_name: str,
    llm_config: dict,
    out_dir: str = "results",
) -> None:
    """
    Save the combined analyzer + explainer outputs to disk.

    Writes two files:
        <out_dir>/<exp_name>_results.json   — full structured output per session
        <out_dir>/<exp_name>_summary.txt    — human-readable plaintext summary

    Each entry in `results` is expected to have:
        session_id, intent, intent_ts, analyzer_output (str), explanation_output (str)
    """
    os.makedirs(out_dir, exist_ok=True)

    model_name = llm_config.get("config_list", [{}])[0].get("model", "unknown")
    saved_at   = datetime.now().isoformat()

    # --- Full JSON output ---
    json_path = os.path.join(out_dir, f"{exp_name}_results.json")
    payload = {
        "experiment": exp_name,
        "model":      model_name,
        "saved_at":   saved_at,
        "sessions":   [],
    }

    for r in results:
        # Try to parse analyzer_output as JSON for cleaner storage;
        # fall back to raw string if the model returned non-JSON.
        try:
            analyzer_parsed = json.loads(r.get("analyzer_output", "{}"))
        except (json.JSONDecodeError, TypeError):
            analyzer_parsed = r.get("analyzer_output", "")

        payload["sessions"].append({
            "session_id":         r.get("session_id"),
            "intent":             r.get("intent"),
            "intent_ts":          r.get("intent_ts"),
            "analyzer_output":    analyzer_parsed,
            "explanation_output": r.get("explanation_output"),
        })

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[save_intent_analysis_results] JSON results → {json_path}")

    # --- Human-readable summary ---
    txt_path = os.path.join(out_dir, f"{exp_name}_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(f"Experiment : {exp_name}\n")
        fh.write(f"Model      : {model_name}\n")
        fh.write(f"Saved at   : {saved_at}\n")
        fh.write(f"Sessions   : {len(results)}\n")
        fh.write("=" * 70 + "\n\n")

        for r in results:
            fh.write(f"Session ID  : {r.get('session_id')}\n")
            fh.write(f"Intent      : {r.get('intent')}\n")
            fh.write(f"Intent time : {r.get('intent_ts')}\n")
            fh.write("-" * 40 + "\n")
            fh.write("Analyzer Output:\n")
            fh.write(r.get("analyzer_output", "") + "\n\n")
            fh.write("Explanation:\n")
            fh.write(r.get("explanation_output", "") + "\n")
            fh.write("=" * 70 + "\n\n")

    print(f"[save_intent_analysis_results] Text summary → {txt_path}")

def read_jsonl_log_file(log_file_path: str) -> list[dict]:
    """
    Read a raw JSONL log file and return a list of parsed event dicts.

    Expects one JSON object per line, e.g.:
        {"data": {...}, "event": "node.created", "ts": "2026-02-23T11:29:56+00:00"}
        {"data": {...}, "event": "workload_version.deployed", "ts": "..."}

    - Blank lines are skipped silently.
    - Malformed lines are skipped with a warning.
    - Events are returned in file order (assumed to be chronological).
    """
    if not os.path.isfile(log_file_path):
        raise FileNotFoundError(f"Log file not found: {log_file_path}")

    log_events = []
    with open(log_file_path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                log_events.append(event)
            except json.JSONDecodeError as e:
                print(f"[Warning] Skipping malformed line {lineno} in '{log_file_path}': {e}")

    print(f"[read_jsonl_log_file] Loaded {len(log_events)} event(s) from '{log_file_path}'")
    return log_events


def save_intent_analysis_results(
    results: list[dict],
    exp_name: str,
    llm_config: dict,
    out_dir: str = "results",
) -> None:
    """
    Save the combined analyzer + explainer outputs to disk.

    Writes two files:
        <out_dir>/<exp_name>_results.json   — full structured output per session
        <out_dir>/<exp_name>_summary.txt    — human-readable plaintext summary

    Each entry in `results` is expected to have:
        session_id, intent, intent_ts, analyzer_output (str), explanation_output (str)
    """
    os.makedirs(out_dir, exist_ok=True)

    model_name = llm_config.get("config_list", [{}])[0].get("model", "unknown")
    saved_at   = datetime.now().isoformat()

    # --- Full JSON output ---
    json_path = os.path.join(out_dir, f"{exp_name}_results.json")
    payload = {
        "experiment": exp_name,
        "model":      model_name,
        "saved_at":   saved_at,
        "sessions":   [],
    }

    for r in results:
        # Try to parse analyzer_output as JSON for cleaner storage;
        # fall back to raw string if the model returned non-JSON.
        try:
            analyzer_parsed = json.loads(r.get("analyzer_output", "{}"))
        except (json.JSONDecodeError, TypeError):
            analyzer_parsed = r.get("analyzer_output", "")

        payload["sessions"].append({
            "session_id":         r.get("session_id"),
            "intent":             r.get("intent"),
            "intent_ts":          r.get("intent_ts"),
            "analyzer_output":    analyzer_parsed,
            "explanation_output": r.get("explanation_output"),
        })

    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    print(f"[save_intent_analysis_results] JSON results → {json_path}")

    # --- Human-readable summary ---
    txt_path = os.path.join(out_dir, f"{exp_name}_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as fh:
        fh.write(f"Experiment : {exp_name}\n")
        fh.write(f"Model      : {model_name}\n")
        fh.write(f"Saved at   : {saved_at}\n")
        fh.write(f"Sessions   : {len(results)}\n")
        fh.write("=" * 70 + "\n\n")

        for r in results:
            fh.write(f"Session ID  : {r.get('session_id')}\n")
            fh.write(f"Intent      : {r.get('intent')}\n")
            fh.write(f"Intent time : {r.get('intent_ts')}\n")
            fh.write("-" * 40 + "\n")
            fh.write("Analyzer Output:\n")
            fh.write(r.get("analyzer_output", "") + "\n\n")
            fh.write("Explanation:\n")
            fh.write(r.get("explanation_output", "") + "\n")
            fh.write("=" * 70 + "\n\n")

    print(f"[save_intent_analysis_results] Text summary → {txt_path}")
