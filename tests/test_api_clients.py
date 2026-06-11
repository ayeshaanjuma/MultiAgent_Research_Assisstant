import unittest
from unittest.mock import patch, AsyncMock, MagicMock
import httpx
import sys
import os

# Ensure the parent directory is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.clients import groq_generate, tavily_search

class TestAPIClients(unittest.IsolatedAsyncioTestCase):
    
    @patch("utils.clients.httpx.AsyncClient")
    async def test_tavily_search_success(self, mock_async_client):
        # Arrange
        mock_client_instance = AsyncMock()
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance
        
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "results": [
                {
                    "title": "Test Title",
                    "url": "https://test.com",
                    "snippet": "This is a test snippet.",
                    "published_date": "2026-06-11",
                    "score": 0.95
                }
            ]
        }
        mock_client_instance.post.return_value = mock_response
        
        # Act
        results = await tavily_search("test query", top_k=1)
        
        # Assert
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["title"], "Test Title")
        self.assertEqual(results[0]["url"], "https://test.com")
        self.assertEqual(results[0]["abstract"], "This is a test snippet.")
        self.assertEqual(results[0]["score"], 0.95)
        mock_client_instance.post.assert_called_once()

    @patch("utils.clients.httpx.AsyncClient")
    async def test_tavily_search_error(self, mock_async_client):
        # Arrange
        mock_client_instance = AsyncMock()
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance
        
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
            "Internal Server Error", request=MagicMock(), response=mock_response
        )
        mock_client_instance.post.return_value = mock_response
        
        # Act & Assert
        with self.assertRaises(httpx.HTTPStatusError):
            await tavily_search("test query")

    @patch("utils.clients.httpx.AsyncClient")
    async def test_groq_generate_success(self, mock_async_client):
        # Arrange
        mock_client_instance = AsyncMock()
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance
        
        mock_response = MagicMock(spec=httpx.Response)
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Groq response text"
                    }
                }
            ]
        }
        mock_client_instance.post.return_value = mock_response
        
        # Act
        response = await groq_generate("Hello Groq")
        
        # Assert
        self.assertEqual(response, "Groq response text")
        # Should call the first URL
        mock_client_instance.post.assert_called_once()

    @patch("utils.clients.httpx.AsyncClient")
    async def test_groq_generate_fallback_success(self, mock_async_client):
        # Arrange
        mock_client_instance = AsyncMock()
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance
        
        # First call fails (400), second call succeeds (200)
        mock_resp_fail = MagicMock(spec=httpx.Response)
        mock_resp_fail.status_code = 400
        mock_resp_fail.text = "Model decommissioned"
        
        mock_resp_success = MagicMock(spec=httpx.Response)
        mock_resp_success.status_code = 200
        mock_resp_success.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Success content"
                    }
                }
            ]
        }
        
        mock_client_instance.post.side_effect = [mock_resp_fail, mock_resp_success]
        
        # Act
        response = await groq_generate("Hello Groq")
        
        # Assert
        self.assertEqual(response, "Success content")
        self.assertEqual(mock_client_instance.post.call_count, 2)

    @patch("utils.clients.httpx.AsyncClient")
    async def test_groq_generate_all_fail(self, mock_async_client):
        # Arrange
        mock_client_instance = AsyncMock()
        mock_async_client.return_value.__aenter__.return_value = mock_client_instance
        
        mock_resp_fail = MagicMock(spec=httpx.Response)
        mock_resp_fail.status_code = 500
        mock_resp_fail.text = "Server Error"
        
        mock_client_instance.post.return_value = mock_resp_fail
        
        # Act & Assert
        with self.assertRaises(httpx.HTTPStatusError):
            await groq_generate("Hello Groq")
        self.assertEqual(mock_client_instance.post.call_count, 3) # Groq has 3 endpoints to try

if __name__ == "__main__":
    unittest.main()
