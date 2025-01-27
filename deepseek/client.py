import requests
import json

class DeepSeekError(Exception):
    """Base exception for all DeepSeek SDK errors."""
    def __init__(self, message):
        super().__init__(message)

class DeepSeekAPIError(DeepSeekError):
    """Exception raised for errors returned by the DeepSeek API."""
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code

class DeepSeekRequestError(DeepSeekError):
    """Exception raised for errors during the HTTP request."""
    def __init__(self, message):
        super().__init__(message)

class DeepSeekClient:
    def __init__(self, api_key, base_url="https://api.deepseek.com/v1"):
        """
        Initialize the DeepSeek client.

        :param api_key: Your DeepSeek API key.
        :param base_url: The base URL for the DeepSeek API (default is the production endpoint).
        """
        self.api_key = api_key
        self.base_url = base_url

    def _make_request(self, endpoint, payload):
        """
        Helper method to make API requests.

        :param endpoint: The API endpoint (e.g., "/chat/completions").
        :param payload: The payload to send in the request.
        :return: The JSON response from the API.
        :raises DeepSeekAPIError: If the API returns an error.
        :raises DeepSeekRequestError: If the HTTP request fails.
        """
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload))
            response.raise_for_status()  # Raise an exception for HTTP errors
            return response.json()
        except requests.exceptions.HTTPError as e:
            # Handle HTTP errors (e.g., 4xx, 5xx)
            raise DeepSeekAPIError(
                f"API request failed with status code {response.status_code}: {response.text}",
                status_code=response.status_code
            )
        except requests.exceptions.RequestException as e:
            # Handle other request-related errors (e.g., network issues)
            raise DeepSeekRequestError(f"Request failed: {e}")

    def generate_chat_completion(self, model, messages, max_tokens=50, temperature=0.7):
        """
        Generate a chat completion using the DeepSeek API.

        :param model: The model to use (e.g., "deepseek-model-name").
        :param messages: A list of message dictionaries (e.g., [{"role": "user", "content": "Hello!"}]).
        :param max_tokens: The maximum number of tokens to generate (default: 50).
        :param temperature: The sampling temperature (default: 0.7).
        :return: The generated text.
        :raises DeepSeekAPIError: If the API returns an error.
        :raises DeepSeekRequestError: If the HTTP request fails.
        """
        endpoint = "/chat/completions"
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature
        }

        response = self._make_request(endpoint, payload)
        if response and "choices" in response:
            return response["choices"][0]["message"]["content"].strip()
        raise DeepSeekAPIError("Invalid response format from the API.")