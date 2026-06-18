import openai
import time
import google.generativeai as genai


def _normalize_messages(user_messages):
    if isinstance(user_messages, (str, dict)):
        return [user_messages]
    return list(user_messages)


class OpenAIClient:
    def __init__(self, model, openai_api_key):
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.model = model
    
    def __call__(self, sys_prompt, user_messages):
        user_content = self._build_user_content(_normalize_messages(user_messages))
        messages = [
            {"role": "developer", "content": sys_prompt},
            {"role": "user", "content": user_content}
        ]
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=1.0,
            )
            result = response.choices[0].message.content
        except Exception:
            time.sleep(10)
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=1.0,
                )
                result = response.choices[0].message.content
            except Exception:
                result = "Something went wrong for this query."
        return result

    @staticmethod
    def _build_user_content(user_messages):
        content = []
        text_buffer = []

        def flush_text():
            if text_buffer:
                content.append({"type": "text", "text": "\n".join(text_buffer)})
                text_buffer.clear()

        for message in user_messages:
            if isinstance(message, dict) and "data" in message:
                flush_text()
                mime_type = message.get("mime_type", "image/png")
                content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{mime_type};base64,{message['data']}"
                    }
                })
            else:
                text_buffer.append(str(message))

        flush_text()
        return content


class GeminiClient:
    def __init__(self, model, gemini_api_key):
        genai.configure(api_key=gemini_api_key)
        self.client = genai.GenerativeModel(model)
        self.model = model
    
    def __call__(self, sys_prompt, user_messages):
        contents = _normalize_messages(user_messages)
        try:
            self.client = genai.GenerativeModel(self.model, system_instruction=sys_prompt)
            response = self.client.generate_content(
                contents=contents,
                generation_config = genai.GenerationConfig(
                    temperature=1.0)
            )
            result = response.text
        except Exception:
            # sometimes the API might fail if we have too many requests
            time.sleep(10)
            try:
                response = self.client.generate_content(
                    contents=contents,
                    generation_config = genai.GenerationConfig(
                        temperature=1.0)
                )
                result = response.text
            except Exception:
                result = "Something went wrong for this query."

        return result
