import os
import json
import re
from typing import List, Dict

from videoagent.clients import GeminiClient, OpenAIClient

from videoagent.logging import get_logger
from videoagent.video_reader import build_video_reader
import videoagent.agent_utils as agent_utils


class VideoAgent:
    def __init__(self, config, api_keys):
        self.config = config
        self.logger = get_logger()

        # Video Reader
        self.video_reader = build_video_reader(config, api_keys)

        # LLM
        if 'openai' in config.actor.lm_option:
            self.llm = OpenAIClient(config.actor.lm_model, self._get_api_key(api_keys, config.actor.lm_option))
        elif 'gemini' in config.actor.lm_option:
            self.llm = GeminiClient(config.actor.lm_model, self._get_api_key(api_keys, config.actor.lm_option))
        else:
            raise ValueError(f"Invalid LLM option: {config.actor.lm_option}")
        
        # VLM
        if 'openai' in config.actor.vlm_option:
            self.vlm = OpenAIClient(config.actor.vlm_model, self._get_api_key(api_keys, config.actor.vlm_option))
        elif 'gemini' in config.actor.vlm_option:
            self.vlm = GeminiClient(config.actor.vlm_model, self._get_api_key(api_keys, config.actor.vlm_option))
        else:
            raise ValueError(f"Invalid VLM option: {config.actor.vlm_option}")

        # video id and qa count
        # for loading and saving intermediate results
        self.video_id = None
        self.video_qa_count = 0
        return

    @staticmethod
    def _get_api_key(api_keys, provider):
        if provider not in api_keys or not api_keys[provider]:
            env_name = "OPENAI_API_KEY" if "openai" in provider else "GEMINI_API_KEY"
            raise ValueError(
                f"Missing API key for provider '{provider}'. Add it to misc/API_KEYS.json "
                f"or set {env_name}."
            )
        return api_keys[provider]

    def execute(self, video_path: str, question: str, captions: List[Dict], subtitles: List[Dict] = None, save_dir: str = None):
        """ Main function to execute the video agent.
        Args:
            video_path: str, the path to the video file.
            question: str, the question to be answered by the video agent.
            captions: List[Dict], the captions of the video.
            subtitles: List[Dict], the subtitles of the video.
            save_dir: str, the directory to save the video agent's output, especially the intermediate step results. save_dir is generally data/{run_name}/{video_id}/
        """
        if save_dir is None:
            save_dir = os.path.join("data", self.video_id or "video")
        os.makedirs(save_dir, exist_ok=True)

        self.video_reader.set_video_path(video_path)

        # ---------- STAGE Analysis 1.1: Map - Segment-level User Intention Analysis ----------
        self.logger.info("[Stage 1.1: Map] - Segment-level User Intention Analysis\n\n")

        # check intermediate results
        segment_user_intention_dir = os.path.join(save_dir, "1_1_user_intention")
        os.makedirs(segment_user_intention_dir, exist_ok=True)
        segment_user_intention_file = os.path.join(segment_user_intention_dir, f"{self.video_qa_count}.json")
        if os.path.exists(segment_user_intention_file):
            with open(segment_user_intention_file, "r") as f:
                segment_user_intentions = json.load(f)
        else:
            segment_user_intentions = self.segment_user_intention_analysis(question, captions, subtitles)
            with open(segment_user_intention_file, "w") as f:
                json.dump(segment_user_intentions, f, indent=2)

        # --------- STAGE Analysis 1.2: Reduce -Video-level User Intention Analysis ---------
        self.logger.info("[Stage 1.2: Reduce] - Video-level User Intention Analysis\n\n")
        video_user_intention_dir = os.path.join(save_dir, "1_2_user_intention")
        os.makedirs(video_user_intention_dir, exist_ok=True)
        video_user_intention_file = os.path.join(video_user_intention_dir, f"{self.video_qa_count}.json")
        if os.path.exists(video_user_intention_file):
            with open(video_user_intention_file, "r") as f:
                video_user_intention = json.load(f)
        else:
            video_user_intention = self.video_user_intention_analysis(question, segment_user_intentions)
            self.logger.info(f"Video-level User Intention Analysis:\n{video_user_intention}\n")
            with open(video_user_intention_file, "w") as f:
                json.dump(video_user_intention, f, indent=2)

        # --------- STAGE Analysis 2.1: Map - Segment-level Goal-aware Analysis ---------
        self.logger.info("[Stage 2.1: Map] - Segment-level Goal-aware Analysis\n\n")
        
        # Step 1: Raising the correct questions for segment-level analysis
        self.logger.info("[Stage 2.1.1: Map] - Proposing Goals\n\n")
        goal_proposal_dir = os.path.join(save_dir, "2_1_1_goal_proposal")
        os.makedirs(goal_proposal_dir, exist_ok=True)
        goal_proposal_file = os.path.join(goal_proposal_dir, f"{self.video_qa_count}.json")
        if os.path.exists(goal_proposal_file):
            with open(goal_proposal_file, "r") as f:
                goal_proposals = json.load(f)
        else:
            goal_proposals = self.goal_proposal(question, video_user_intention)
            self.logger.info(f"Goal Proposal:\n{goal_proposals['response']}\n")
            with open(goal_proposal_file, "w") as f:
                json.dump(goal_proposals, f, indent=2)

        # Step 2: Calling the VLM for segment-level goal-aware analysis
        self.logger.info(f"[Stage 2.1.2: Map] - Calling VLMs for Segment-level Goal-aware Analysis\n\n")
        local_question = goal_proposals['local_question']
        global_question = goal_proposals['global_question']

        segment_list = agent_utils.extract_segment_list_from_video_level_intentions(video_user_intention)
        perception_results_dir = os.path.join(save_dir, "2_1_2_perception_results")
        os.makedirs(perception_results_dir, exist_ok=True)
        perception_results_file = os.path.join(perception_results_dir, f"{self.video_qa_count}.json")
        if os.path.exists(perception_results_file):
            with open(perception_results_file, "r") as f:
                perception_results = json.load(f)
        else:
            self.logger.info(f"Relevant Segments: {segment_list}")

            # global question for the reasoning on the whole video
            global_results = self.goal_aware_analysis_global(question, global_question, segment_list, max_fps=2.0, max_frame_num=32)

            # local question for the perception on each segment
            local_results = self.goal_aware_analysis_local(question, local_question, segment_list, max_fps=2.0, max_frame_num=32)

            perception_results = {
                "local_results": local_results,
                "global_results": global_results
            }
            with open(perception_results_file, "w") as f:
                json.dump(perception_results, f, indent=2)

        # --------- STAGE Analysis 2.2: Reduce - Video-level Goal-aware Analysis ---------
        self.logger.info("[Stage 2.2: Reduce] - Video-level Goal-aware Analysis\n\n")
        answer_generation_dir = os.path.join(save_dir, "2_2_answer_generation")
        os.makedirs(answer_generation_dir, exist_ok=True)
        answer_generation_file = os.path.join(answer_generation_dir, f"{self.video_qa_count}.json")
        if os.path.exists(answer_generation_file):
            with open(answer_generation_file, "r") as f:
                answer_generation = json.load(f)
                letter_grade = answer_generation['letter_grade']
                answer = answer_generation['answer']
        else:
            answer, letter_grade = self.answer_generation(question, video_user_intention, goal_proposals, perception_results)
            answer_generation = {
                "answer": answer,
                "letter_grade": letter_grade
            }
            with open(answer_generation_file, "w") as f:
                json.dump(answer_generation, f, indent=2)

        self.logger.info(f"Answer Generation:\n{answer}")
        return answer_generation
    
    def segment_user_intention_analysis(self, question, captions, subtitles=None, num_scene_per_segment=32):
        """ Segment-level User Intention Analysis.
            TODO: Due to limited resources, using for-loop temporarily.
        """
        scene_cnt = 0
        segment_intentions_list = []
        while scene_cnt < len(captions):
            segment_captions = captions[scene_cnt:min(scene_cnt+num_scene_per_segment, len(captions))]
            segment_captions_text = agent_utils.intention_analysis_caption_to_text(segment_captions)

            import videoagent.prompts.intention_analysis as intention_analysis_prompts
            sys_prompt = intention_analysis_prompts.SYS_PROMPT

            # 1. question
            question_str = f'[1. Question]: {question}'

            # 2. video frames
            question_str += '\n [2. Video Frames]:\n'
            question_str += f'Video Frames are sampled between {segment_captions[0]["t_start"]:.1f} to {segment_captions[-1]["t_end"]:.1f} seconds from a video with total duration of {self.video_reader.video_duration:.1f} seconds.\n'
            frames = []
            for caption in segment_captions:
                t_start, t_end = caption['t_start'], caption['t_end']

                # read video frames at the center of each scene
                read_t_start = (t_start + t_end) / 2
                read_t_end = read_t_start + 0.5
                frame = self.video_reader.read(read_t_start, read_t_end, fps=1.0)
                frames.append({'data': agent_utils.img_to_base64(frame[0]), 'mime_type': 'image/png'})
            
            # 3. Captions
            caption_str = f'[3. Captions]:\n'
            caption_str += segment_captions_text

            # combine into user prompts
            user_messages = [intention_analysis_prompts.SEGMENT_INTENTION_ANALYSIS] + [question_str] + frames + [caption_str]

            # call vlm
            intentions = self.vlm(sys_prompt, user_messages)

            # save intentions
            segment_intentions = {
                "t_start": segment_captions[0]['t_start'],
                "t_end": segment_captions[-1]['t_end'],
                "intentions": intentions
            }
            segment_intentions_list.append(segment_intentions)

            # update scene_cnt
            scene_cnt += num_scene_per_segment

            # logging
            t_start, t_end = segment_captions[0]['t_start'], segment_captions[-1]['t_end']
            self.logger.info(f"Segment Intention Analysis for ({t_start:.2f} - {t_end:.2f}):\n{intentions}\n\n")

        return segment_intentions_list

    def video_user_intention_analysis(self, question, segment_intentions_list):
        """ Video-level User Intention Analysis.
        """
        import videoagent.prompts.intention_analysis as intention_analysis_prompts
        sys_prompt = intention_analysis_prompts.SYS_PROMPT
        
        question_str = f'[1. Question]: {question}\n\n'

        segment_intention_analyses_str = "[2. Segment Intention Analysis]:\n"
        segment_intention_analyses_str += f'The total duration of the video is {self.video_reader.video_duration:.1f} seconds.\n'
        for segment_intention in segment_intentions_list:
            t_start, t_end = segment_intention['t_start'], segment_intention['t_end']
            segment_intention_analyses_str += f'----------\nAnalysis for Video Segment ({t_start:.1f} - {t_end:.1f}):\n----------\n'
            segment_intention_analyses_str += f"{segment_intention['intentions']}\n\n"

        user_messages = [intention_analysis_prompts.VIDEO_INTENTION_ANALYSIS] + [question_str] + [segment_intention_analyses_str]
        video_user_intention = self.llm(sys_prompt, user_messages)
        return video_user_intention

    def goal_proposal(self, question, video_user_intention):
        """ Raising the correct questions for segment-level analysis.
        """
        import videoagent.prompts.goal_aware_analysis as goal_aware_analysis_prompts
        sys_prompt = goal_aware_analysis_prompts.SYS_PROMPT

        # 1. question
        question_str = f'[1. Question]: {question}'

        # 2. video-level user intention
        question_str += '\n [2. Video Context Analysis]:\n'
        question_str += f"{video_user_intention}"

        user_messages = [goal_aware_analysis_prompts.GOAL_PROPOSAL_PROMPT] + [question_str]
        goal_proposal_response = self.llm(sys_prompt, user_messages)

        # extract local and global questions
        def extract_questions(text):
            # Find the indices of the section markers
            local_start = text.find('[2. Local Question]')
            global_start = text.find('[3. Global Question]')
            
            local_question = None
            global_question = None
            
            if local_start != -1 and global_start != -1:
                # Extract local question (between the two markers)
                local_content_start = text.find(':', local_start) + 1  # Start after the colon
                if local_content_start != 0:  # Make sure colon was found
                    local_question = text[local_content_start:global_start].strip()
            
            if global_start != -1:
                # Extract global question (everything after the global marker)
                global_content_start = text.find(':', global_start) + 1  # Start after the colon
                if global_content_start != 0:  # Make sure colon was found
                    global_question = text[global_content_start:].strip()
            
            return local_question, global_question
        
        local_question, global_question = extract_questions(goal_proposal_response)

        if local_question is None:
            local_question = question
            self.logger.warning("Could not parse local question from goal proposal; falling back to the original question.")
        if global_question is None:
            global_question = question
            self.logger.warning("Could not parse global question from goal proposal; falling back to the original question.")

        goal_proposal = {
            "response": goal_proposal_response,
            "local_question": local_question,
            "global_question": global_question
        }
        return goal_proposal

    def goal_aware_analysis_local(self, question, local_question, segment_list, max_fps=2.0, max_frame_num=32):
        """ Detailed analysis by sampling densely within each segment.
        """
        import videoagent.prompts.goal_aware_analysis as goal_aware_analysis_prompts
        vlm_response_list = []
        for segment in segment_list:
            t_start, t_end = segment

            # sample the frames
            local_fps = max_fps
            if int(max_fps * (t_end - t_start)) > max_frame_num:
                # decrease the fps
                local_fps = max_frame_num / (t_end - t_start)
                frames = self.video_reader.read(t_start, t_end, fps=local_fps)
            else:
                frames = self.video_reader.read(t_start, t_end, fps=local_fps)
            
            # user messages
            sys_prompt = goal_aware_analysis_prompts.SYS_PROMPT
            question_str = goal_aware_analysis_prompts.CALL_VLM_PROMPT
            question_str += f'[1. Original Question]: {question}\n\n [2. Target Question]: {local_question}\n\n [3. Video Frames]:\n'
            
            frames_str = []
            for frame in frames:
                frames_str.append({'data': agent_utils.img_to_base64(frame), 'mime_type': 'image/png'})
            
            user_messages = [question_str] + frames_str

            # call vlm
            vlm_response = self.vlm(sys_prompt, user_messages)

            vlm_response_list.append({
                "t_start": t_start,
                "t_end": t_end,
                "response": vlm_response
            })

            # logging
            self.logger.info(f"Local VLM Response for Segment ({t_start:.2f} - {t_end:.2f}):\n{vlm_response}")

        return vlm_response_list

    def goal_aware_analysis_global(self, question, global_question, segment_list, max_fps=2.0, max_frame_num=32):
        """ Global analysis by sampling sparsely across segments
        """
        import videoagent.prompts.goal_aware_analysis as goal_aware_analysis_prompts
        sys_prompt = goal_aware_analysis_prompts.SYS_PROMPT
        vlm_response_list = []

        # 1. A global sparse sampling as baseline
        frames, timestamps = agent_utils.sample_video_frames_uniform(self.video_reader.video_path, frame_num=max_frame_num)
        question_str = goal_aware_analysis_prompts.CALL_VLM_PROMPT
        question_str += f'[1. Original Question]: {question}\n\n [2. Target Question]: {global_question}\n\n [3. Video Frames]:\n'
            
        frames_str = []
        for frame in frames:
            frames_str.append({'data': agent_utils.img_to_base64(frame), 'mime_type': 'image/png'})
        
        user_messages = [question_str] + frames_str

        # call vlm
        vlm_response = self.vlm(sys_prompt, user_messages)

        vlm_response_list.append({
            "timestamp": timestamps,
            "response": vlm_response
        })

        timestamp_str = ", ".join([f"{ts:.1f}" for ts in timestamps])
        self.logger.info(f"Global VLM Response for [{timestamp_str}]:\n{vlm_response}\n\n")

        # 2. Iteratively conducting cross-segment analysis
        if len(segment_list) > 0:
            # many many scenes, let's sample a single frame per segment
            if len(segment_list) > max_frame_num:
                segment_cnt = 0
                while segment_cnt < len(segment_list):
                    segment_chunk = segment_list[segment_cnt:min(segment_cnt + max_frame_num, len(segment_list))]
                    frame_pils, timestamps = [], []
                    for segment in segment_chunk:
                        t_start, t_end = segment
                        timestamp = (t_start + t_end) / 2
                        frame_pils += self.video_reader.read(timestamp, timestamp + 0.5, fps=1.0)
                        timestamps.append(timestamp)
                    frames_str = []
                    for frame_pil in frame_pils:
                        frames_str.append({'data': agent_utils.img_to_base64(frame_pil), 'mime_type': 'image/png'})
                    
                    user_messages = [question_str] + frames_str
    
                    # call vlm
                    vlm_response = self.vlm(sys_prompt, user_messages)
    
                    vlm_response_list.append({
                        "timestamp": timestamps,
                        "response": vlm_response
                    })
                    timestamp_str = ", ".join([f"{ts:.1f}" for ts in timestamps])
                    self.logger.info(f"Global VLM Response for [{timestamp_str}]:\n{vlm_response}\n\n")
    
                    segment_cnt += max_frame_num
                return vlm_response_list
            
            # 3. If less than max_frame_num, we can perceive more frames per segment
            num_frame_per_segment = max_frame_num // len(segment_list)
            frame_pils, timestamps = [], []
            for segment in segment_list:
                t_start, t_end = segment
    
                for ts in range(num_frame_per_segment):
                    timestamp = t_start * (num_frame_per_segment - ts) / (num_frame_per_segment + 1) + t_end * (ts + 1) / (num_frame_per_segment + 1)
                    frame_pils += self.video_reader.read(timestamp, timestamp + 0.5, fps=1.0)
                    timestamps.append(timestamp)
                
            frames_str = []
            for frame_pil in frame_pils:
                frames_str.append({'data': agent_utils.img_to_base64(frame_pil), 'mime_type': 'image/png'})
            
            user_messages = [question_str] + frames_str
            # call vlm
            vlm_response = self.vlm(sys_prompt, user_messages)
            vlm_response_list.append({
                "timestamp": timestamps,
                "response": vlm_response
            })
            timestamp_str = ", ".join([f"{ts:.1f}" for ts in timestamps])
            self.logger.info(f"Global VLM Response for [{timestamp_str}]:\n{vlm_response}\n\n")
        return vlm_response_list

    def answer_generation(self, question, video_user_intention, goal_proposals, perception_results):
        """ Answer generation.
        """
        import videoagent.prompts.goal_aware_analysis as goal_aware_analysis_prompts
        sys_prompt = goal_aware_analysis_prompts.SYS_PROMPT

        prompts = goal_aware_analysis_prompts.ANSWER_GENERATION_PROMPT
        prompts += f'The total duration of the video is {self.video_reader.video_duration:.1f} seconds.\n'
        prompts += f"\n\n# 1. User Question\n {question}\n\n"
        
        prompts += f"\n\n# 2. Video-level User Intention Analysis\n{video_user_intention}\n\n"

        prompts += f"\n\n# 3. Goal Proposal\n{goal_proposals['response']}\n\n"

        prompts += "\n\n# 4. Goal-aware Analysis\n"
        prompts += "## 4.1 Local Analysis\n"
        for result in perception_results['local_results']:
            t_start, t_end = result['t_start'], result['t_end']
            response = result['response']
            prompts += f"Analysis for Video Segment ({t_start:.1f} - {t_end:.1f}):\n{response}\n\n"

        prompts += "\n\n## 4.2 Global Analysis\n"
        for result in perception_results['global_results']:
            timestamps = result['timestamp']
            response = result['response']

            timestamps_str = ", ".join([f"{ts:.1f}" for ts in timestamps])
            prompts += f"Global Analysis with Frames at {timestamps_str}:\n{response}\n\n"
        
        prompts += "\n\n Please start your response below.\n"
        answer = self.llm(sys_prompt, [prompts])

        letter_grade = self.answer_separation(answer)
        return answer, letter_grade
    
    def answer_separation(self, answer):
        answer_match = re.search(r"\[2\.\s*Answer\]\s*:\s*([A-D])\b", answer, flags=re.IGNORECASE)
        if answer_match:
            return answer_match.group(1).upper()

        prompt = "Please read the following responses and extract the answer A, B, C, or D from it inside the [2. Answer] section. Only return a single letter from A to D and nothing else."
        sys_prompt = "You are a helpful assistant."
        user_messages = [prompt] + [answer]
        answer = self.llm(sys_prompt, user_messages)
        fallback_match = re.search(r"\b([A-D])\b", answer, flags=re.IGNORECASE)
        if fallback_match:
            return fallback_match.group(1).upper()
        return answer
