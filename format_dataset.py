#!/usr/bin/env python3
"""
make_openrlhf_dataset.py
Convert data/base_dataset.json -> data/openrlhf/train.jsonl and valid.jsonl
"""

import argparse, json, pathlib, itertools
from training.utils import serialize_tilde


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/base_dataset.json")
    p.add_argument("--prompt", default="agent/system_prompt.txt")
    p.add_argument("--out_dir", default="data/openrlhf")
    args = p.parse_args()

    # --- load source files ---------------------------------------------------
    src  = json.loads(pathlib.Path(args.input).read_text(encoding="utf-8"))
    sys_prompt = pathlib.Path(args.prompt).read_text(encoding="utf-8").strip()

    # Collect all records first
    all_records = []
    for example in src:
        # Many datasets store a *list* of paraphrased questions ─ flatten it
        questions = example["question"]
        if isinstance(questions, (list, tuple)):
            iterator = questions
        else:
            iterator = [questions]

        for q in iterator:
            # Gather label data for serialization
            label_data = [example["answer"], "retrieval"]
            record = {
                "context_messages": [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user",   "content": q}
                ],
                "label": serialize_tilde(label_data)
            }
            all_records.append(record)

    # Split the records: 90% train, 10% validation
    total_records = len(all_records)
    train_size = int(total_records * 0.9)
    
    train_records = all_records[:train_size]
    valid_records = all_records[train_size:]

    # Create output directory
    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    
    # Write train.jsonl
    train_path = pathlib.Path(args.out_dir) / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as fh:
        for record in train_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    
    # Write valid.jsonl
    valid_path = pathlib.Path(args.out_dir) / "valid.jsonl"
    with valid_path.open("w", encoding="utf-8") as fh:
        for record in valid_records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✓ wrote {train_path} ({len(train_records)} records)")
    print(f"✓ wrote {valid_path} ({len(valid_records)} records)")

if __name__ == "__main__":
    main()