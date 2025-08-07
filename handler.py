"""GPT-OSS 120B RunPod Serverless Handler"""

import runpod
import asyncio
import httpx
import logging
from typing import Dict, Any, Optional
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
GPT_OSS_API_URL = os.getenv("GPT_OSS_API_URL", "http://localhost:8000")
GPT_OSS_API_KEY = os.getenv("GPT_OSS_API_KEY", "")
DEFAULT_MAX_TOKENS = int(os.getenv("DEFAULT_MAX_TOKENS", "2048"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

# Global HTTP client for reuse
http_client: Optional[httpx.AsyncClient] = None

def initialize_client():
    """Initialize HTTP client"""
    global http_client
    if http_client is None:
        http_client = httpx.AsyncClient(timeout=300.0)
        logger.info("HTTP client initialized")

async def call_gpt_oss_api(messages, max_tokens=None, temperature=None, model="gpt-oss-120b"):
    """Call GPT-OSS API"""
    if http_client is None:
        initialize_client()
    
    # Prepare request
    request_data = {
        "messages": messages,
        "model": model,
        "max_tokens": max_tokens or DEFAULT_MAX_TOKENS,
        "temperature": temperature or DEFAULT_TEMPERATURE,
        "stream": False
    }
    
    # Prepare headers
    headers = {"Content-Type": "application/json"}
    if GPT_OSS_API_KEY:
        headers["Authorization"] = f"Bearer {GPT_OSS_API_KEY}"
    
    try:
        logger.info(f"Making request to GPT-OSS API: {GPT_OSS_API_URL}")
        response = await http_client.post(
            f"{GPT_OSS_API_URL}/v1/chat/completions",
            json=request_data,
            headers=headers
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info("Successfully received response from GPT-OSS API")
            return result
        else:
            error_msg = f"GPT-OSS API error: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise Exception(error_msg)
            
    except httpx.RequestError as e:
        error_msg = f"Failed to connect to GPT-OSS API: {str(e)}"
        logger.error(error_msg)
        raise Exception(error_msg)

def handler(job):
    """
    RunPod serverless handler function
    
    Expected input format:
    {
        "input": {
            "messages": [
                {"role": "user", "content": "Hello World"}
            ],
            "max_tokens": 100,
            "temperature": 0.7,
            "model": "gpt-oss-120b"
        }
    }
    """
    try:
        logger.info(f"Received job: {job}")
        
        # Extract input from job
        job_input = job.get("input", {})
        
        # Support both simple prompt and messages format
        if "prompt" in job_input:
            # Simple format: {"input": {"prompt": "Hello World"}}
            messages = [{"role": "user", "content": job_input["prompt"]}]
        elif "messages" in job_input:
            # OpenAI format: {"input": {"messages": [{"role": "user", "content": "Hello"}]}}
            messages = job_input["messages"]
        else:
            return {
                "error": "Missing required field: either 'prompt' or 'messages' must be provided",
                "status": "error"
            }
        
        # Validate messages format
        if not isinstance(messages, list) or len(messages) == 0:
            return {
                "error": "Messages must be a non-empty list",
                "status": "error"
            }
        
        # Extract other parameters
        max_tokens = job_input.get("max_tokens")
        temperature = job_input.get("temperature")
        model = job_input.get("model", "gpt-oss-120b")
        
        # Check if GPT-OSS API is configured
        if not GPT_OSS_API_URL or GPT_OSS_API_URL == "http://localhost:8000":
            # Return mock response for testing
            logger.warning("GPT-OSS API not configured, returning mock response")
            return {
                "id": "chatcmpl-mock-123",
                "object": "chat.completion",
                "created": 1640995200,
                "model": model,
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": f"Mock response: I received your message '{messages[-1].get('content', '')}'. This is a test response from GPT-OSS 120B serverless worker!"
                        },
                        "finish_reason": "stop"
                    }
                ],
                "usage": {
                    "prompt_tokens": 20,
                    "completion_tokens": 30,
                    "total_tokens": 50
                },
                "status": "success"
            }
        
        # Process the request asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            result = loop.run_until_complete(
                call_gpt_oss_api(messages, max_tokens, temperature, model)
            )
            result["status"] = "success"
            return result
            
        finally:
            loop.close()
            
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "status": "error"
        }

# Initialize client on startup
initialize_client()

# Start the RunPod serverless worker
if __name__ == "__main__":
    logger.info("Starting GPT-OSS 120B RunPod Serverless Worker")
    runpod.serverless.start({"handler": handler})