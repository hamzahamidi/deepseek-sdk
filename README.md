# DeepSeek Python SDK

A Python SDK for interacting with the DeepSeek API.

## Installation

```bash
pip install deepseek-sdk
```

## Usage

```python
from deepseek import DeepSeekClient, DeepSeekAPIError, DeepSeekRequestError

# Initialize the client
client = DeepSeekClient(api_key="your_deepseek_api_key")

try:
    # Generate a chat completion
    quote = client.generate_chat_completion(
        model="deepseek-model-name",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Provide a motivational quote."}
        ]
    )
    print(quote)
except DeepSeekAPIError as e:
    print(f"API Error: {e}")
except DeepSeekRequestError as e:
    print(f"Request Error: {e}")
```
