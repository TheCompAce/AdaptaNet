import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch

from modules.llms.OpenAI import OpenAI

class LLM:
    def __init__(self, model_name='stabilityai/StableBeluga-7B', max_length=1000, use_causal_pretrained=True, trust_remote_code=False, use_cache=True, use_api=False, api_key=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.use_cache = use_cache
        self.use_api = use_api
        self.api_key = api_key
        self.cache_file = f"data/{__name__}_cache.json"
        self.openai = OpenAI(api_key=api_key) if use_api else None

        if use_causal_pretrained:
            self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif not use_causal_pretrained and trust_remote_code:
            self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(self.device)
        else:
            self.model = None
            self.tokenizer = None

        if self.use_cache:
            if not os.path.exists('data'):
                os.makedirs('data')
            if not os.path.isfile(self.cache_file):
                with open(self.cache_file, 'w') as file:
                    json.dump({}, file)

    def generate_response(self, system_input, user_input):
        if self.use_cache:
            response = self.check_cache(system_input, user_input)
            if response:
                return response

        if self.use_api:
            api_response = self.openai.ask(self.model, system_input, user_input)
            return api_response['choices'][0]['message']['content']

        # Non-API response generation
        prompt = f"### System:\n{system_input}\n### User:\n{user_input}\n### Assistant:\n"
        inputs = self.tokenizer.encode(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = inputs.to(self.device)
        outputs = self.model.generate(inputs, max_length=self.max_length, num_return_sequences=1)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("### Assistant:\n")[-1]

        if self.use_cache:
            self.update_cache(system_input, user_input, response)

        return response

    def get_embeddings(self, text, max_length=8192):
        # Encode the text and apply mean pooling
        inputs = self.model.encode(text, max_length=max_length, return_tensors='pt')
        return inputs.mean(dim=1)

    def check_cache(self, system_input, user_input):
        with open(self.cache_file, 'r') as file:
            cache = json.load(file)
        return cache.get((system_input, user_input))

    def update_cache(self, system_input, user_input, response):
        with open(self.cache_file, 'r') as file:
            cache = json.load(file)
        cache[(system_input, user_input)] = response
        with open(self.cache_file, 'w') as file:
            json.dump(cache, file)

# Example usage
system_input = "You are a math expert assistant. Your mission is to help users understand and solve various math problems."
user_input = "calculate 100 + 520 + 60"
api_key = os.getenv('OPENAI_API_KEY')  # Fetch API key from environment variable
llm = LLM(use_cache=True, use_api=True, api_key=api_key)
response = llm.generate_response(system_input, user_input)
print(response)