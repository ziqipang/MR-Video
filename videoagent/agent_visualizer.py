import os
import json
import cv2
import re
from copy import deepcopy
from typing import List, Dict, Tuple
from PIL import Image

from videoagent.clients import GeminiClient, OpenAIClient

from videoagent.logging import get_logger
from videoagent.video_reader import build_video_reader
import videoagent.agent_utils as agent_utils


def second_to_timestamp(t):
    minute = int(t // 60)
    second = t - minute * 60
    return f"{minute}:{second:.2f}"


class VideoAgentVisualizer:
    """ Visualize the reasoning process of the video agent.
    """
    def __init__(self, config, api_keys):
        self.config = config

        # video id and qa count
        # for loading and saving intermediate results
        self.video_id = None
        self.video_qa_count = 0
        return
    
    def visualize(self, video_path: str, question: str, captions: List[Dict], subtitles: List[Dict] = None, save_dir: str = None,):
        print("Visualizing the video agent's reasoning process...")

        # Stage 1.1: Segment-level User Intention Analysis
        print("--------------------------------")
        print("[Stage 1.1: Map] - Segment-level User Intention Analysis\n\n")
        print("--------------------------------")
        segment_user_intention_dir = os.path.join(save_dir, "1_1_user_intention")
        segment_user_intention_file = os.path.join(segment_user_intention_dir, f"{self.video_qa_count}.json")
        with open(segment_user_intention_file, "r") as f:
            segment_user_intentions = json.load(f)
        
        for segment in segment_user_intentions:
            t_start, t_end = segment['t_start'], segment['t_end']
            intentions = segment['intentions']
            print(f"Segment Intention Analysis for ({second_to_timestamp(t_start)} - {second_to_timestamp(t_end)}): {intentions}\n\n")
        
        # Stage 1.2: Video-level User Intention Analysis
        print("--------------------------------")
        print("[Stage 1.2: Reduce] - Video-level User Intention Analysis")
        print("--------------------------------")
        video_user_intention_dir = os.path.join(save_dir, "1_2_user_intention")
        video_user_intention_file = os.path.join(video_user_intention_dir, f"{self.video_qa_count}.json")
        with open(video_user_intention_file, "r") as f:
            video_user_intention = json.load(f)
        
        print(f"Video-level User Intention Analysis: {video_user_intention}\n\n")
        print("--------------------------------")
        segment_list = agent_utils.extract_segment_list_from_video_level_intentions(video_user_intention)
        print("Extracted Segments:")
        print(segment_list)
        print("--------------------------------")

        # Stage 2.1: Segment-level Goal-aware Analysis
        print("--------------------------------")
        print("[Stage 2.1: Map] - Segment-level Goal-aware Analysis\n\n")
        print("--------------------------------")
        print("[Stage 2.1.1: Map] - Proposing Goals\n\n")
        goal_proposal_dir = os.path.join(save_dir, "2_1_1_goal_proposal")
        goal_proposal_file = os.path.join(goal_proposal_dir, f"{self.video_qa_count}.json")
        with open(goal_proposal_file, "r") as f:
            goal_proposals = json.load(f)
        
        print(f"Goal Proposal: {goal_proposals['response']}\n\n")

        print("[Stage 2.1.2: Map] - Calling VLMs for Segment-level Goal-aware Analysis\n\n")
        local_question = goal_proposals['local_question']
        global_question = goal_proposals['global_question']

        perception_results_dir = os.path.join(save_dir, "2_1_2_perception_results")
        perception_results_file = os.path.join(perception_results_dir, f"{self.video_qa_count}.json")
        with open(perception_results_file, "r") as f:
            perception_results = json.load(f)
        
        local_results = perception_results['local_results']
        global_results = perception_results['global_results']

        print("--------------------------------")
        print("Local Results:")
        for result in local_results:
            print(f"Segment ({second_to_timestamp(result['t_start'])} - {second_to_timestamp(result['t_end'])}): {result['response']}\n\n")
        print("--------------------------------")

        print("--------------------------------")
        print("Global Results:")
        for result in global_results:
            timestamps = result['timestamp']
            response = result['response']
            timestamp_str = ", ".join([second_to_timestamp(ts) for ts in timestamps])
            print(f"Segment ({timestamp_str}): {response}\n\n")
        print("--------------------------------")

        # Stage 2.2: Answer Generation
        print("--------------------------------")
        print("[Stage 2.2: Reduce] - Answer Generation\n\n")
        print("--------------------------------")
        answer_generation_dir = os.path.join(save_dir, "2_2_answer_generation")
        answer_generation_file = os.path.join(answer_generation_dir, f"{self.video_qa_count}.json")
        with open(answer_generation_file, "r") as f:
            answer_generation = json.load(f)
        
        print(f"Answer Generation: {answer_generation['answer']}\n\n")
        print(f"Letter Grade: {answer_generation['letter_grade']}\n\n")
        print("--------------------------------")
        return
