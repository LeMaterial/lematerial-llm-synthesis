from mistralai import Mistral
from openai import OpenAI, AzureOpenAI
from cohere import ClientV2
import os

class LLM: 

    def __init__(self, model_name: str, provider: str, port: int = 8000):
        self.model_name = model_name
        self.provider = provider

        if self.provider == "mistral":
            self.client = Mistral(api_key=os.getenv("MISTRAL_API_KEY"))
        elif self.provider == "cohere":
            self.client = ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        elif self.provider == "openai":
            endpoint = "https://gpt-amayuelas.openai.azure.com/"
            subscription_key = os.getenv("OPENAI_API_KEY")
            api_version = "2024-12-01-preview"
            self.client = AzureOpenAI(api_version=api_version, azure_endpoint=endpoint, api_key=subscription_key)
        elif self.provider == "vllm":
            self.client = OpenAI(base_url=f"http://localhost:{port}/v1", api_key=os.getenv("VLLM_API_KEY"))
        

    def generate_text(self, prompt: str, response_format: str = None):

        if self.provider == "mistral":
            response = self.client.chat.complete(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                response_format=response_format
            )
            return response.choices[0].message.content
        
        elif self.provider == "vllm":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        
        elif self.provider == "cohere":
            response = self.client.chat(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.message.content[0].text
        
        elif self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.choices[0].message.content
        else:
            raise ValueError(f"Provider {self.provider} not supported")
    