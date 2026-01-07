#!/usr/bin/env python3
import re
import os
import sys
from collections import defaultdict

# Regex to capture Hadoop/HDFS block IDs (blk_ followed by digits or minus)
BLOCK_RE = re.compile(r"(blk_[\-0-9]+)")

def extract_sessions(input_file, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    sessions = defaultdict(list)

    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            # Find all block IDs in the line (usually one)
            matches = BLOCK_RE.findall(line)
            if matches:
                # If a line contains multiple block_ids, duplicate line to each session
                for blk_id in matches:
                    sessions[blk_id].append(line)
            else:
                # Optional: collect lines without block IDs (ignore by default)
                pass

    # Write sessions to files
    for blk_id, lines in sessions.items():
        out_path = os.path.join(output_dir, f"{blk_id}.log")
        with open(out_path, "w", encoding="utf-8") as out:
            out.writelines(lines)

    print(f"Created {len(sessions)} session log files in {output_dir}")


def main():
    if len(sys.argv) != 3:
        print("Usage: python split_sessions_by_block.py <raw_log_file_path> <output_dir_path>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    extract_sessions(input_file, output_dir)

if __name__ == "__main__":
    main()
