import base64
import os
import requests

class OpenAI:
    def __init__(self, api_key, api_url="https://api.openai.com/v1/chat/completions"):
        self.api_key = api_key
        self.api_url = api_url

    def ask(self, model, system_input, user_input):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        data = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_input},
                {"role": "user", "content": user_input}
            ]
        }
        response = requests.post(self.api_url, json=data, headers=headers)
        print(f"response = {response}")
        return response.json()

    def ask_with_image(self, text_input, image, model="gpt-4-vision-preview", max_tokens=8012):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        image_data = self.prepare_image(image)
        data = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_input},
                        image_data
                    ]
                }
            ],
            "max_tokens": max_tokens
        }
        response = requests.post(self.api_url, json=data, headers=headers)
        return response.json()

    def prepare_image(self, image):
        if self.is_url(image):
            return {"type": "image_url", "image_url": {"url": image}}
        elif os.path.isfile(image):
            with open(image, "rb") as image_file:
                return {"type": "image", "data": base64.b64encode(image_file.read()).decode('utf-8')}
        else:
            # Assuming image is already byte data
            return {"type": "image", "data": base64.b64encode(image).decode('utf-8')}

    def is_url(self, string):
        return string.startswith('http://') or string.startswith('https://')

    def create_speech(self, text, model='tts-1', voice='alloy', response_format='mp3', speed=1):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "input": text,
            "voice": voice,
            "response_format": response_format,
            "speed": speed
        }
        response = requests.post(f"{self.api_url}/audio/speech", json=data, headers=headers)
        return response.content

    def transcribe_audio(self, audio_file, model='whisper-1', language=None, prompt=None, response_format='json', temperature=0):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        files = {
            "file": audio_file,
            "model": (None, model),
            "language": (None, language),
            "prompt": (None, prompt),
            "response_format": (None, response_format),
            "temperature": (None, str(temperature))
        }
        response = requests.post(f"{self.api_url}/audio/transcriptions", files=files, headers=headers)
        return response.json()

    def translate_audio(self, audio_file, model='whisper-1', prompt=None, response_format='json', temperature=0):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        files = {
            "file": audio_file,
            "model": (None, model),
            "prompt": (None, prompt),
            "response_format": (None, response_format),
            "temperature": (None, str(temperature))
        }
        response = requests.post(f"{self.api_url}/audio/translations", files=files, headers=headers)
        return response.json()
    
    def create_embeddings(self, input_text, model="text-embedding-ada-002", encoding_format="float"):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "input": input_text,
            "model": model,
            "encoding_format": encoding_format
        }
        response = requests.post(f"{self.api_url}/embeddings", json=data, headers=headers)
        return response.json()
    
    def create_image(self, prompt, model="dall-e-3", n=1, quality="standard", response_format="url", size="1024x1024", style="vivid"):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "quality": quality,
            "response_format": response_format,
            "size": size,
            "style": style
        }
        response = requests.post(f"{self.api_url}/images/generations", json=data, headers=headers)
        return response.json()

    def create_image_edit(self, image_file, prompt, mask_file=None, model="dall-e-3", n=1, size="1024x1024", response_format="url"):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        files = {
            "image": image_file,
            "prompt": (None, prompt),
            "mask": mask_file,
            "model": (None, model),
            "n": (None, str(n)),
            "size": (None, size),
            "response_format": (None, response_format)
        }
        response = requests.post(f"{self.api_url}/images/edits", files=files, headers=headers)
        return response.json()

    def create_image_variation(self, image_file, model="dall-e-3", n=1, size="1024x1024", response_format="url"):
        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        files = {
            "image": image_file,
            "model": (None, model),
            "n": (None, str(n)),
            "size": (None, size),
            "response_format": (None, response_format)
        }
        response = requests.post(f"{self.api_url}/images/variations", files=files, headers=headers)
        return response.json()