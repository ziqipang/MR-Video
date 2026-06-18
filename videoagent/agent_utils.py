from io import BytesIO
import base64
import re
import cv2
from PIL import Image


def intention_analysis_caption_to_text(captions):
    """ Converting the captions to text.
    """
    result = ""
    for caption in captions:
        t_start, t_end = caption["t_start"], caption["t_end"]
        detailed_description = caption["detailed_description"]
        result += f"[{t_start:.1f} - {t_end:.1f}] {detailed_description}\n"
    return result


def img_to_base64(img):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    return img_str


def extract_segment_list_from_video_level_intentions(intention: str):
    # candidate_scenes_pattern = r"\[2\. Relevant Segments\]:([\s\S]*?)(?=\[3\. Key Characters\]:)"
    candidate_scenes_pattern = r"\[2\. Relevant Segments\]:\s*([\s\S]*?)(?=\[3\. Key Characters\]:|\Z)"
    candidate_scenes_match = re.search(candidate_scenes_pattern, intention)
        
    if candidate_scenes_match:
        candidate_scenes_block = candidate_scenes_match.group(1)
        # Regular expression to match all (t_start, t_end) tuples
        tuple_pattern = r"\(\s*(\d+(?:\.\d+)?)\s*[-,]\s*(\d+(?:\.\d+)?)\s*\)"
        tuples = re.findall(tuple_pattern, candidate_scenes_block)
        # Convert to a list of float tuples
        candidate_scenes = [(float(start), float(end)) for start, end in tuples]
    else:
        candidate_scenes = []
    
    if len(candidate_scenes) > 0:
        return candidate_scenes
    
    candidate_scenes_pattern = r"\[2\. Candidate Scenes\]:([\s\S]*?)(?=\[3\. Key Characters\]:)"
    candidate_scenes_match = re.search(candidate_scenes_pattern, intention)
    if candidate_scenes_match:
        candidate_scenes_block = candidate_scenes_match.group(1)
        number_pairs_pattern = r'\[(\d+(?:\.\d+)?),\s*(\d+(?:\.\d+)?)\]'
        number_pairs = re.findall(number_pairs_pattern, candidate_scenes_block)
        candidate_scenes = [[float(start), float(end)] for start, end in number_pairs]
    else:
        candidate_scenes = []
    return candidate_scenes


def sample_video_frames_uniform(video_path, frame_num=128):
    """ Using video frames
    """
    video = cv2.VideoCapture(video_path)
    video_fps = video.get(cv2.CAP_PROP_FPS)
    if not video.isOpened() or video_fps <= 0:
        raise ValueError(f"Could not read video: {video_path}")

    video_duration = video.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
    current_t_range = [0, video_duration]
    t_start = int(current_t_range[0] * video_fps)
    t_end = int(current_t_range[1] * video_fps)
    freq = max(1, int(t_end / max(1, frame_num - 1)))

    timestamps = []
    frames = []
    for t in range(t_start, t_end, freq):
        video.set(cv2.CAP_PROP_POS_FRAMES, t)
        ret, frame = video.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb = cv2.resize(frame_rgb, (360 * frame_rgb.shape[1] // frame_rgb.shape[0], 360), interpolation=cv2.INTER_LANCZOS4)
            frame_pil = Image.fromarray(frame_rgb)
            frames.append(frame_pil)
            timestamps.append(t / video_fps)
    return frames, timestamps
