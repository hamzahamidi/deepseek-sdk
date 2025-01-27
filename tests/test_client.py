import unittest
import requests
from deepseek import DeepSeekClient, DeepSeekAPIError, DeepSeekRequestError
from unittest.mock import patch, MagicMock

class TestDeepSeekClient(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_api_key"
        self.client = DeepSeekClient(self.api_key)

    @patch("requests.post")
    def test_generate_chat_completion_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test response"}}]
        }
        mock_post.return_value = mock_response
        
        messages = [{"role": "user", "content": "Hello!"}]
        response = self.client.generate_chat_completion("test-model", messages)
        
        self.assertEqual(response, "Test response")
        mock_post.assert_called_once()

    @patch("requests.post")
    def test_generate_chat_completion_api_error(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.text = "Bad Request"
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError()
        mock_post.return_value = mock_response

        messages = [{"role": "user", "content": "Hello!"}]
        with self.assertRaises(DeepSeekAPIError):
            self.client.generate_chat_completion("test-model", messages)

    @patch("requests.post")
    def test_generate_chat_completion_request_error(self, mock_post):
        mock_post.side_effect = requests.exceptions.RequestException("Network error")
        messages = [{"role": "user", "content": "Hello!"}]
        with self.assertRaises(DeepSeekRequestError):
            self.client.generate_chat_completion("test-model", messages)

if __name__ == "__main__":
    unittest.main()
