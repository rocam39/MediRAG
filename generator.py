import os
import requests


class LLMGenerator:

    def __init__(self):

        self.api_url = (
            "https://router.huggingface.co/v1/chat/completions"
        )

        self.api_key = os.getenv("HF_TOKEN")

        if not self.api_key:
            raise ValueError(
                "HF_TOKEN is not configured."
            )

        self.model_name = "openai/gpt-oss-120b:fastest"

    def generate(self, query, context):

        prompt = f"""
You are a medical knowledge assistant.

Answer the user's question using ONLY the
information provided in the sources below.

Rules:

1. Do not use outside knowledge.
2. Do not invent medical facts.
3. If the sources do not contain enough
   information to answer the question, say:
   "I couldn't find this information in the
   provided documents."
4. Keep the answer concise and clear.
5. Cite claims using the source labels provided,
   for example [Source 1].
6. Do not provide a diagnosis or personalized
   medical advice.

SOURCES:

{context}

QUESTION:

{query}

ANSWER:
"""

        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful medical "
                        "knowledge assistant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 300,
            "temperature": 0.2
        }

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        response = requests.post(
            self.api_url,
            headers=headers,
            json=payload,
            timeout=120
        )

        if response.status_code != 200:
            raise RuntimeError(
                f"LLM API error {response.status_code}: "
                f"{response.text}"
            )

        data = response.json()

        return data["choices"][0]["message"]["content"].strip()