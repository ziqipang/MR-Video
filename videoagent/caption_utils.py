import re


def extract_captions(captions):
    results = []
    for caption in captions:
        t_start, t_end = caption['t_start'], caption['t_end']
        caption = caption['caption']

        brief_desc_pattern = r"\[1\. Brief Description\]: (.*?)\n\n"
        detailed_desc_pattern = r"\[2\. Detailed Description\]: (.*)"

        try:
            brief_description = re.search(brief_desc_pattern, caption, re.DOTALL)
            detailed_description = re.search(detailed_desc_pattern, caption, re.DOTALL)
            brief_description = brief_description.group(1).strip() if brief_description else ""
            detailed_description = detailed_description.group(1).strip() if detailed_description else ""
        except:
            brief_description = ""
            detailed_description = ""

        results.append({
            "t_start": t_start,
            "t_end": t_end,
            "brief_description": brief_description,
            "detailed_description": detailed_description,
        })
    return results


def extract_captions_vis_mem(captions):
    results = []
    for caption in captions:
        t_start, t_end = caption['t_start'], caption['t_end']
        caption = caption['caption']

        brief_desc_pattern = r"\[1\. Brief Description\]: (.*?)\n\n"
        characters_pattern = r"\[2\. Appeared Characters\]:\s*\[([^\]]+)\]: (.*?)\n"
        detailed_desc_pattern = r"\[3\. Detailed Description\]: (.*)"

        try:
            brief_description = re.search(brief_desc_pattern, caption, re.DOTALL)
            detailed_description = re.search(detailed_desc_pattern, caption, re.DOTALL)
            brief_description = brief_description.group(1).strip() if brief_description else ""
            detailed_description = detailed_description.group(1).strip() if detailed_description else ""
        except:
            brief_description = ""
            detailed_description = ""

        results.append({
            "t_start": t_start,
            "t_end": t_end,
            "brief_description": brief_description,
            "detailed_description": detailed_description,
        })
    return results


def insert_subtitle_into_captions(captions, subtitles):
    for i, caption in enumerate(captions):
        t_start, t_end = caption['t_start'], caption['t_end']
        cur_segment_subtitles = []
        for sub in subtitles:
            subtitle_start, subtitle_end = sub['t_start'], sub['t_end']
            # convert to seconds
            if (subtitle_start >= t_start and subtitle_start <= t_end) or (subtitle_end >= t_start and subtitle_end <= t_end):
                cur_segment_subtitles.append({
                    "t_start": subtitle_start,
                    "t_end": subtitle_end,
                    "text": sub['text'],
                })
        
        subtitle_str = f'''[Subtitles]:\n'''
        if len(cur_segment_subtitles) > 0:
            for sub in cur_segment_subtitles:
                subtitle_str += f'({sub["t_start"]:.1f}, {sub["t_end"]:.1f}): {sub["text"]}\n'
            caption['detailed_description'] += f'\n\n{subtitle_str}'
        else:
            subtitle_str += f'No subtitles for this segment.'
            caption['detailed_description'] += f'\n\n{subtitle_str}'
    return captions


def insert_subtitle_into_captions_json(captions, subtitles):
    for i, caption in enumerate(captions):
        t_start, t_end = caption['t_start'], caption['t_end']
        cur_segment_subtitles = []
        for sub in subtitles:
            subtitle_start, subtitle_end = time_to_seconds(sub['start']), time_to_seconds(sub['end'])
            # convert to seconds
            if (subtitle_start >= t_start and subtitle_start <= t_end) or (subtitle_end >= t_start and subtitle_end <= t_end):
                cur_segment_subtitles.append({
                    "t_start": subtitle_start,
                    "t_end": subtitle_end,
                    "text": sub['line'],
                })
        
        subtitle_str = f'''[Subtitles]:\n'''
        if len(cur_segment_subtitles) > 0:
            for sub in cur_segment_subtitles:
                subtitle_str += f'[{sub["t_start"]:.1f}, {sub["t_end"]:.1f}]: {sub["text"]}\n'
            caption['detailed_description'] += f'\n\n{subtitle_str}'
        else:
            subtitle_str += f'No subtitles for this segment.'
            caption['detailed_description'] += f'\n\n{subtitle_str}'
    return captions


def time_to_seconds(time_str):
    # Split by colon to get hours, minutes, and seconds (with milliseconds)
    parts = time_str.split(':')
    
    # Extract hours, minutes, and seconds
    hours = int(parts[0])
    minutes = int(parts[1])
    seconds = float(parts[2])
    
    # Convert everything to seconds
    total_seconds = hours * 3600 + minutes * 60 + seconds
    
    return total_seconds
