import argparse
import json
import os
from pathlib import Path

from omegaconf import OmegaConf

from videoagent.agent import VideoAgent
from videoagent.caption_utils import extract_captions_vis_mem
from videoagent.config_utils import load_api_keys
from videoagent.logging import setup_logger


def env_default(name, default):
    return os.environ.get(name, default)


def load_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def select_contents(contents, video_ids=None, limit=None, num_processes=1, process_index=0):
    if video_ids:
        allowed = set(video_ids)
        contents = [content for content in contents if content["video_id"] in allowed]

    if limit is not None:
        contents = contents[:limit]

    if num_processes > 1:
        if process_index < 0 or process_index >= num_processes:
            raise ValueError("--process_index must be in [0, --num_processes)")
        contents = contents[process_index::num_processes]

    return contents


def main(args):
    logger = setup_logger(
        name="egoschema",
        log_file=os.path.join(args.log_dir, f"{args.exp_name}.log"),
    )
    logger.info(f"Logging to {os.path.join(args.log_dir, f'{args.exp_name}.log')}")

    config = OmegaConf.load(args.config)
    egoschema_contents = select_contents(
        load_jsonl(args.egoschema_meta),
        video_ids=args.video_ids,
        limit=args.limit,
        num_processes=args.num_processes,
        process_index=args.process_index,
    )

    api_keys = load_api_keys(args.api_key)
    agent = VideoAgent(config=config, api_keys=api_keys)

    output_dir = Path(args.output_dir) / args.exp_name
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Running {len(egoschema_contents)} EgoSchema videos")
    for content in egoschema_contents:
        video_id = content["video_id"]
        video_path = Path(args.egoschema_path) / f"{video_id}.mp4"
        if not video_path.exists():
            logger.warning(f"Skipping {video_id}: video file not found at {video_path}")
            continue

        caption_path = Path(args.caption_dir) / f"{video_id}.json"
        if not caption_path.exists():
            logger.warning(f"Skipping {video_id}: caption file not found at {caption_path}")
            continue

        with open(caption_path, "r") as f:
            captions = extract_captions_vis_mem(json.load(f))

        agent.video_id = video_id
        video_output_dir = output_dir / video_id
        video_output_dir.mkdir(parents=True, exist_ok=True)

        for qa_index, qa in enumerate(content["qas"]):
            question = qa["question"]
            answer = qa["answer"]
            options = qa["options"]
            question_with_context = (
                f"{question}\n"
                "The video is ego-centric. The character c in the questions represents the first-person perspective of the character.\n"
                f"{options}"
            )
            logger.info(f"Start video {video_id}, qa {qa_index + 1}/{len(content['qas'])}")
            logger.info(f"Question: {question}\n{options}")
            logger.info(f"Answer: {answer}")

            agent.video_qa_count = qa_index
            answer_generation = agent.execute(
                str(video_path),
                question_with_context,
                captions,
                save_dir=str(video_output_dir),
            )
            letter_grade = answer_generation["letter_grade"].upper()

            logger.info(f"Prediction: {letter_grade}")
            logger.info(f"Correct: {letter_grade == answer}")

        logger.info(f"Finished video {video_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MR. Video on EgoSchema.")
    parser.add_argument("--egoschema_path", type=str, default=env_default("EGOSCHEMA_PATH", "datasets/egoschema/videos"))
    parser.add_argument("--egoschema_meta", type=str, default=env_default("EGOSCHEMA_META", "datasets/egoschema/egoschema_val_data.jsonl"))
    parser.add_argument("--caption_dir", type=str, default=env_default("EGOSCHEMA_CAPTION_DIR", "datasets/egoschema/captions"))
    parser.add_argument("--config", type=str, default=env_default("MRVIDEO_CONFIG", "configs/gpt4o_gemini.yaml"))
    parser.add_argument("--api_key", type=str, default=env_default("MRVIDEO_API_KEYS", "misc/API_KEYS.json"))
    parser.add_argument("--output_dir", type=str, default=env_default("MRVIDEO_OUTPUT_DIR", "data/egoschema"))
    parser.add_argument("--exp_name", type=str, default=env_default("MRVIDEO_EXP_NAME", "mrvideo_gpt4o_gemini"))
    parser.add_argument("--log_dir", type=str, default=env_default("MRVIDEO_LOG_DIR", "logs/egoschema"))
    parser.add_argument("--video_ids", nargs="*", default=None, help="Optional video IDs for a targeted run.")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit after filtering.")
    parser.add_argument("--num_processes", type=int, default=int(env_default("MRVIDEO_NUM_PROCESSES", "1")))
    parser.add_argument("--process_index", type=int, default=int(env_default("MRVIDEO_PROCESS_INDEX", "0")))
    main(parser.parse_args())
