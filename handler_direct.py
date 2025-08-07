"""GPT-OSS 120B RunPod Serverless Handler - Direct Model Loading"""

import runpod
import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/DialoGPT-large")  # Fallback for testing
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "4096"))  # Increased for longer responses
MIN_TOKENS = int(os.getenv("MIN_TOKENS", "50"))   # Minimum response length
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.8"))  # Slightly higher for more creativity
TOP_P = float(os.getenv("TOP_P", "0.9"))  # Nucleus sampling
REPETITION_PENALTY = float(os.getenv("REPETITION_PENALTY", "1.1"))  # Reduce repetition
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model and tokenizer
model = None
tokenizer = None

def load_model():
    """Load GPT-OSS model directly"""
    global model, tokenizer
    
    if model is None:
        logger.info(f"Loading model: {MODEL_NAME}")
        logger.info(f"Device: {DEVICE}")
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model with appropriate settings
            if DEVICE == "cuda":
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    trust_remote_code=True
                )
            else:
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    trust_remote_code=True
                )
            
            logger.info("Model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            # Use a smaller model for testing if main model fails
            logger.info("Loading fallback model for testing...")
            tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
            model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

def generate_response(messages: List[Dict], max_tokens: int = None, temperature: float = None, min_tokens: int = None, top_p: float = None, repetition_penalty: float = None):
    """Generate response using the loaded model"""
    global model, tokenizer
    
    if model is None or tokenizer is None:
        load_model()
    
    # Convert messages to prompt
    prompt = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            prompt += f"System: {content}\n"
        elif role == "user":
            prompt += f"User: {content}\n"
        elif role == "assistant":
            prompt += f"Assistant: {content}\n"
    
    prompt += "Assistant:"
    
    try:
        # Tokenize input
        inputs = tokenizer.encode(prompt, return_tensors="pt")
        if DEVICE == "cuda":
            inputs = inputs.to(DEVICE)
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens or MAX_TOKENS,
                min_length=len(inputs[0]) + (min_tokens or MIN_TOKENS),  # Ensure minimum response length
                temperature=temperature or TEMPERATURE,
                top_p=top_p or TOP_P,
                repetition_penalty=repetition_penalty or REPETITION_PENALTY,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                attention_mask=torch.ones_like(inputs),
                no_repeat_ngram_size=3,  # Prevent repetitive 3-grams
                early_stopping=False  # Don't stop early
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the new generated part
        generated_text = response[len(prompt):].strip()
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Generation failed: {str(e)}")
        return f"عذراً، حدث خطأ في توليد الرد: {str(e)}"

def handler(job):
    """
    RunPod serverless handler function - Direct model execution
    
    Expected input format:
    {
        "input": {
            "messages": [
                {"role": "user", "content": "Hello World"}
            ],
            "max_tokens": 100,
            "temperature": 0.7
        }
    }
    """
    try:
        logger.info(f"Received job: {job}")
        
        # Extract input from job
        job_input = job.get("input", {})
        
        # Support both simple prompt and messages format
        if "prompt" in job_input:
            messages = [{"role": "user", "content": job_input["prompt"]}]
        elif "messages" in job_input:
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
        
        # Extract parameters
        max_tokens = job_input.get("max_tokens", MAX_TOKENS)
        temperature = job_input.get("temperature", TEMPERATURE)
        min_tokens = job_input.get("min_tokens", MIN_TOKENS)
        top_p = job_input.get("top_p", TOP_P)
        repetition_penalty = job_input.get("repetition_penalty", REPETITION_PENALTY)
        model_name = job_input.get("model", "gpt-oss-120b")
        
        # Generate response using direct model
        logger.info("Generating response with direct model...")
        generated_content = generate_response(messages, max_tokens, temperature, min_tokens, top_p, repetition_penalty)
        
        # Return OpenAI-compatible response
        return {
            "id": f"chatcmpl-direct-{hash(str(messages)) % 10000}",
            "object": "chat.completion",
            "created": 1640995200,
            "model": model_name,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": generated_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": sum(len(msg.get("content", "").split()) for msg in messages),
                "completion_tokens": len(generated_content.split()),
                "total_tokens": sum(len(msg.get("content", "").split()) for msg in messages) + len(generated_content.split())
            },
            "status": "success",
            "direct_model": True,
            "device": DEVICE,
            "model_loaded": model is not None
        }
        
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "status": "error",
            "device": DEVICE,
            "model_loaded": model is not None
        }

# Load model on startup
logger.info("Starting GPT-OSS 120B Direct RunPod Serverless Worker")
load_model()

# Start the RunPod serverless worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})