import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


def env_default(name, default):
    return os.environ.get(name, default)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def main(args):
    videomme_contents = load_jsonl(args.videomme_meta)
    result_dir = Path(args.output_dir) / args.exp_name

    total_qa_num = 0
    right_num = 0
    category_right = defaultdict(int)
    category_total = defaultdict(int)

    for content in videomme_contents:
        video_id = content["video_id"]
        if args.video_ids and video_id not in args.video_ids:
            continue

        for qa_index, qa in enumerate(content["qas"]):
            prediction_path = result_dir / video_id / "2_2_answer_generation" / f"{qa_index}.json"
            if not prediction_path.exists():
                continue

            with open(prediction_path, "r") as f:
                prediction = json.load(f)
            pred_answer = prediction["letter_grade"].upper()
            answer = qa["answer"]
            qa_type = qa["task_type"]

            category_total[qa_type] += 1
            if answer == pred_answer:
                category_right[qa_type] += 1
                right_num += 1
            total_qa_num += 1

    if total_qa_num == 0:
        print(f"No predictions found under {result_dir}")
        return

    for category in sorted(category_total):
        acc = category_right[category] / category_total[category]
        print(f"{category}: {acc:.3f} from {category_total[category]}")
    print(f"Overall: {right_num / total_qa_num:.3f} from {total_qa_num}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate MR. Video predictions on VideoMME.")
    parser.add_argument("--videomme_meta", type=str, default=env_default("VIDEOMME_META", "datasets/videomme/video_mme_long.jsonl"))
    parser.add_argument("--output_dir", type=str, default=env_default("MRVIDEO_OUTPUT_DIR", "data/videomme"))
    parser.add_argument("--exp_name", type=str, default=env_default("MRVIDEO_EXP_NAME", "mrvideo_gpt4o_gemini"))
    parser.add_argument("--video_ids", nargs="*", default=None, help="Optional video IDs to evaluate.")
    main(parser.parse_args())
