import base64
from io import BytesIO
import cv2
from PIL import Image
from videoagent.clients import GeminiClient, OpenAIClient


def build_video_reader(config, api_keys):
    vlm_option, vlm_model = config.video_reader.vlm_option, config.video_reader.vlm_model
    if 'openai' in vlm_option:
        vlm_client = OpenAIClient(vlm_model, _get_api_key(api_keys, vlm_option))
    elif 'gemini' in vlm_option:
        vlm_client = GeminiClient(vlm_model, _get_api_key(api_keys, vlm_option))
    else:
        raise ValueError(f"Invalid video reader: {vlm_option} and {vlm_model}")
    return VideoReader(vlm_client)


def _get_api_key(api_keys, provider):
    if provider not in api_keys or not api_keys[provider]:
        env_name = "OPENAI_API_KEY" if "openai" in provider else "GEMINI_API_KEY"
        raise ValueError(
            f"Missing API key for provider '{provider}'. Add it to misc/API_KEYS.json "
            f"or set {env_name}."
        )
    return api_keys[provider]


class VideoReader:
    """ A Python class containing a video displaying a series of information.
    """
    def __init__(self, vlm_client, max_frames=32):
        """ Initialize the VideoReader with the video_probe, a vision language model.
        """
        self.vlm_client = vlm_client
        self.env_logs = []
        self.vis_memory = None
        self.max_frames = max_frames
    
    def set_video_path(self, video_path):
        self.video_path = video_path
        self.video = cv2.VideoCapture(video_path)
        video_fps = self.video.get(cv2.CAP_PROP_FPS)
        if not self.video.isOpened() or video_fps <= 0:
            raise ValueError(f"Could not read video: {video_path}")
        self.video_duration = self.video.get(cv2.CAP_PROP_FRAME_COUNT) / video_fps
        self.current_t_range = [0, self.video_duration]
    
    def read(self, t_start, t_end, fps=2.0, cap_limit=30):
        """ Read the video from t_start to t_end with a frequency of fps, and return a list of images.
        """
        frames = []
        video_fps = self.video.get(cv2.CAP_PROP_FPS)
        t_start = int(t_start * video_fps)
        t_end = int(t_end * video_fps)
        freq = max(1, int(video_fps / fps))

        while (t_end - t_start) / freq > cap_limit:
            freq = freq * 1.5
        freq = int(freq)
        for t in range(t_start, t_end, freq):
            self.video.set(cv2.CAP_PROP_POS_FRAMES, t)
            ret, frame = self.video.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = cv2.resize(frame_rgb, (360 * frame_rgb.shape[1] // frame_rgb.shape[0], 360), interpolation=cv2.INTER_LANCZOS4)
                frame_pil = Image.fromarray(frame_rgb)
                frames.append(frame_pil)
        
        self.current_t_range = [t_start, t_end]
        return frames

    def query(self, sys_prompt, query_prompt, t_start, t_end, fps=2.0):
        """ Using the vlm_client to answer the question about a few frames of the video selected by the read function.
        """
        frames = self.read(t_start, t_end, fps)

        contents = []
        if not isinstance(frames, list):
            frames = [frames]
        
        for frame in frames:
            buffered = BytesIO()
            frame.save(buffered, format="PNG")
            contents.append({
                "mime_type": "image/png",
                "data": base64.b64encode(buffered.getvalue()).decode("utf-8")
            })
        
        if len(contents) > self.max_frames:
            ratio = int(len(contents) / self.max_frames) + 1
            contents = contents[::ratio]
        
        contents.append(query_prompt)
        result = self.vlm_client(sys_prompt, contents)

        self.env_logs += ['''
        ---
        Analyzing video from {t_start} to {t_end} with a frequency of {fps} fps.
        Question: {query_prompt}
        Response: {result}
        '''.strip().format(t_start=t_start, t_end=t_end, fps=fps, query_prompt=query_prompt, result=result)]
        return result
