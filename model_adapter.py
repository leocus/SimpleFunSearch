import json
import requests
import numpy as np


class Model:
    """An interface for querying an LLM via http requests."""

    def __init__(self, model_url, model_name, key, system_prompt):
        """
        Initialize the interface with the model.
        Params:
            model_url: url for connecting to the model
            model_name: name of the model
            key: API key
            system_prompt: system prompt
        """
        self._model_url = model_url
        self._model_name = model_name
        self._system_prompt = system_prompt
        self._headers = {
            'Content-Type': 'application/json',
            'accept': 'application/json',
            'Authorization': key,
        }

    def get_output(self, prompt):
        """Query the model and get the output."""
        json_data = {
            'messages': [
                {
                    "role": "system",
                    "content": self._system_prompt
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            'params': {
                'top_k': 2,
                'temperature': 1
            },
            'top_k': 2,
            'temperature': 1
        }
        json_data = {
            "n_predict": 256,
            "temperature": 1.28,
            "stop": [],
            "repeat_last_n": 256,
            "repeat_penalty": 1.18,
            "top_k": 40,
            "top_p": 0.95,
            "min_p": 0.05,
            "tfs_z": 1,
            "typical_p": 1,
            "presence_penalty": 0,
            "frequency_penalty": 0,
            "mirostat": 0,
            "mirostat_tau": 5,
            "mirostat_eta": 0.1,
            "grammar": "",
            "n_probs": 0,
            "image_data": [],
            "cache_prompt": False,
            "api_key": "",
            "slot_id": 0,
            "seed": np.random.randint(0, 2**32),
            "prompt": f"{self._system_prompt}<|question|>{prompt}<|question_end|>"
        }

        response = requests.post(self._model_url, headers=self._headers, json=json_data)
        response = json.loads(response.text)
        response = response['content']
        return response

    def __call__(self, prompt):
        """Query the model and get the output."""
        return self.get_output(prompt)
