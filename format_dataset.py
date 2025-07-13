#!/usr/bin/env python3
"""
make_openrlhf_dataset.py
Convert data/base_dataset.json -> data/openrlhf/train.jsonl
"""

import argparse, json, pathlib, itertools

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",  default="data/base_dataset.json")
    p.add_argument("--prompt", default="agent/system_prompt.txt")
    p.add_argument("--out",    default="data/openrlhf/train.jsonl")
    args = p.parse_args()

    # --- load source files ---------------------------------------------------
    src  = json.loads(pathlib.Path(args.input).read_text(encoding="utf-8"))
    sys_prompt = pathlib.Path(args.prompt).read_text(encoding="utf-8").strip()

    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with pathlib.Path(args.out).open("w", encoding="utf-8") as fh:
        for example in src:
            # Many datasets store a *list* of paraphrased questions ─ flatten it
            questions = example["question"]
            if isinstance(questions, (list, tuple)):
                iterator = questions
            else:
                iterator = [questions]

            for q in iterator:
                label_dict = {
                    "task": "retrieval",
                    "answer": example["answer"],
                    "mem_id": "berlin-1"
                }
                record = {
                    "context_messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user",   "content": q}
                    ],
                    "label": json.dumps(label_dict)
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✓ wrote {args.out}")

if __name__ == "__main__":
    main()