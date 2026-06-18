SYS_PROMPT = '''
You are a helpful and advanced AI assistant with the ability of analyzing videos with your computer vision and reasoning abilities.
'''.strip()

GOAL_PROPOSAL_PROMPT = '''
# 1. Motivation

In this step of long video understanding, you are making preparations for calling vision-language models to analyze sampled video frames. Specifically, you will be given the user's question and a video-level analysis from yourself. Based on such information, you will **propose a question to prompt the vision-language models** to analyze the video frames.

You will access the following information:
1. A question.
2. A video-level analysis from yourself. It contains the following information:

1. Reasoning: ... (Your explanation about which parts of the video are relevant to the question.)

2. Relevant Segments: [(t_start, t_end), ...]... (The periods that are potentially relevant from your analysis. Time is represented in seconds.)

3. Key Characters: [(character symnonym in question, identifiable properties or <NAME> in captions), ...]... (The key characters that are mentioned in the question and how to identify them. Keep the list empty if the question is not related to any characters.)

4. Local or Global: ... (Whether the question requires combining contexts from different segments to answer. If "yes", then this is a global question. If "no", then this is a local question.)

# 2. Instructions and Guidelines

## Objectives

When thinking about the questions to ask, please pay attention to how the next step will sample the video frames for your questions. In practice, we will use two ways:
1. Local: Sample N video frames for each relevant segment, e.g., 32 frames. In this way, the vision-language models can use your question to check the details of each segment.
2. Global: Sample 1 video frame for each segment, sequentially. In this way, the vision-language models can use your question to check the flow of the plots or conduct reasoning over a longer period of time.

Therefore, you should propose two questions:
1. A local question: what kind of detailed information or evidence should the vision-language models find in each segment?
2. A global question: what kind of reasoning should the vision-language models conduct on a longer time span?

## Output Formats

Please strictly follow the output formats below to propose your questions, so that the downstream parts can easily extract the information:

[1. Reasoning]: ... (Your reasoning process. Please be precise, concise, and clear. Explicitly thinking about what kind of information is missing or important for the question.)
[2. Local Question]: ... (Your question for the local analysis.)
[3. Global Question]: ... (Your question for the global analysis.)

## Principles and Tips

1. Think carefully about how a short video segment could contribute to long video understanding by paying attention to the question and video segment contents. Some examples are:
    
    - For question on visual details, you should check if the video segment **contains the scene that the user wants**.
    
    - For question on information over a period of time, such as the order or the number of actions, you should reason **whether this segment can contribute part of the analysis**.
    
    - For question on the reason or implication of the story/actions in the video, you should check if the video segment **contains the key information** that can help you understand the story/actions.

2. Keep your question concise, clear, and within a few sentences. Do not enumerate or explicitly depending on any time information.

3. Remember to use the options from the original questions, expressed with (A), (B), (C), (D), to think about the best way to distinguish the correct one. It is also important to include the original options as the context for the vision-language models.

4. Use your knowledge of prompting large language models or vision-language models to improve your question.

5. Your output questions should only contain a question and options. Do not include any analyses, speculations, or reasoning into the question. For example, the question should directly start as "Describe ... (A) ..., (B) ..., (C) ..., (D) ...", "What is ... (A) ..., (B) ..., (C) ..., (D) ..." or similar formats.


# 3. Your Job Starts

'''.strip()

CALL_VLM_PROMPT = '''
You will be given sampled frames from a video to answer a question related to the video. Your input includes:
1. The original question (for you to understand the purpose of the whole analysis).
2. The sampled frames.
3. The target question (the actual question you will be answering based on the frames).

Please pay attention to the following tips:

1. Please state your confidence level when answering the question between 1 (no evidence or information related to the question is presented) and 5 (confident).

2. If the question is a multiple-choice question, please provide reasoning and explanation before providing the answer.

3. Please carefully understand the relationship between the original question and the target question. Most importantly, considering how to find the correct option from the original question.

Directly provide your response after the following information:

'''.strip()

ANSWER_GENERATION_PROMPT = '''
# 1. Motivation

You are at the last step of long video understanding. You will have the user's question and a series of your analysis to finally answer the user's question.

Before conducting actual analysis, it is important to understand the steps of the previoous analysis that will be presented to you:

1. Video-level User Intention Analysis: You first analyze which parts of the video and what kind of characters are relevant to the user's question. You also think about how each video segment could contribute to the long video understanding.

2. Goal Proposal: To call vision-language models to analyze the video segments, you have proposed two questions for the VLMs to use. The first question is called "local question", used for detailed analysis for each segment, and the second question is called "global question", used for joint analysis and reasoning across multiple segments.

3. Goal-aware Analysis: You will receive the results of the vision-language models' perception for each video segment using the local question and across multiple segments using the global question.

By understanding the previous steps, you will have a good understanding of the meaning of the information provided to you, especially which parts are reliable and informative for answering the user's question.

# 2. Instructions and Guidelines

1. Think carefully about how a short video segment could contribute to long video understanding by paying attention to the question and video segment contents. Some examples are:
    
    - For question on visual details, you should check if the video segment **contains the scene that the user wants**.
    
    - For question on information over a period of time, such as the order or the number of actions, you should reason **whether this segment can contribute part of the analysis**.
    
    - For question on the reason or implication of the story/actions in the video, you should check if the video segment **contains the key information** that can help you understand the story/actions.

2. Carefully consider whether the analysis at local segments or across multiple segments is more important for answering the user's question.

3. With the information provided to you, imagine youself as a human watching the video. Figuring out the flow of the plots is critical.

4. It is possible that some information is vague or contradicting each other. You should utilize advanced reasoning skills to resolve the contradictions. Some very useful principles are:
    
    - If the user has mentioned a specific criteria, try to use it **precisely** and **rigorously** in your analysis.
    
    - Try to utilize the confidence levels provided in the answers.
    
    - Always thinking about your strategy: how the analysis at local segments or across multiple segments could contribute to the long video understanding. For example, do you combine the pieces of information together, summing some numbers, or picking the best segment to answer the question?
    
    - Humans have a limited memory. Always prioritize the most salient information.

5. Pay attention to the time information. They might provide additional correspondence information across different segments and analyses.

# 3. Output Format

Please provide your answer in the following format:

[1. Reasoning]: ... (Your advanced reasoning based on the information above.)

[2. Answer]: A captal letter from A to D (If you cannot find a correct answer, please make a guess from A to D based on the information you have. To ensure correct post-processing, please strictly use this format. Do not add any characters or spaces.)

# 4. Your Job Starts

'''.strip()