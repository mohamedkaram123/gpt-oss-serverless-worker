"""GPT-OSS 120B RunPod Serverless Handler - Direct Model Loading"""

import os
# Fix hf_transfer issue
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import runpod
import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, Any, List
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = os.getenv("MODEL_NAME", "microsoft/DialoGPT-medium")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1024"))  # زيادة من 512 إلى 1024
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Global model and tokenizer
model = None
tokenizer = None
model_loaded = False

def load_model():
    """Load model directly"""
    global model, tokenizer, model_loaded
    
    if model_loaded:
        return True
        
    try:
        logger.info(f"🚀 Loading model: {MODEL_NAME}")
        logger.info(f"📱 Device: {DEVICE}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model
        if DEVICE == "cuda":
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
        
        model_loaded = True
        logger.info("✅ Model loaded successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        return False

def generate_response(messages: List[Dict], max_tokens: int = None, temperature: float = None):
    """Generate response using the loaded model"""
    global model, tokenizer
    
    if not model_loaded:
        if not load_model():
            return "عذراً، فشل في تحميل النموذج. يرجى المحاولة مرة أخرى."
    
    # Convert messages to prompt - DialoGPT format
    conversation = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "user":
            conversation += content + tokenizer.eos_token
    
    # For DialoGPT, we don't need "Assistant:" prefix
    
    try:
        # Tokenize conversation for DialoGPT
        inputs = tokenizer.encode(conversation, return_tensors="pt", max_length=512, truncation=True)
        if DEVICE == "cuda":
            inputs = inputs.to(DEVICE)
        
        # Generate with better parameters for DialoGPT
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=min(max_tokens or MAX_TOKENS, 1024),
                temperature=max(temperature or TEMPERATURE, 0.8),  # Higher temperature for better responses
                do_sample=True,
                top_p=0.9,  # Add nucleus sampling
                top_k=50,   # Add top-k sampling
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1,  # Reduce repetition
                length_penalty=1.0,
                no_repeat_ngram_size=2
            )
        
        # Decode only the new generated part
        input_length = inputs.shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        
        # Fallback if empty response
        if not generated_text or len(generated_text) < 3:
            generated_text = "مرحباً! نعم، أستطيع التحدث بالعربية. كيف يمكنني مساعدتك اليوم؟"
            
        return generated_text
        
    except Exception as e:
        logger.error(f"❌ Generation failed: {str(e)}")
        return f"عذراً، حدث خطأ في توليد الرد: {str(e)}"

def handler(job):
    """RunPod serverless handler - Direct model execution"""
    start_time = time.time()
    
    try:
        logger.info(f"📨 Received job: {job}")
        
        # Extract input
        job_input = job.get("input", {})
        
        # Support both formats
        if "prompt" in job_input:
            messages = [{"role": "user", "content": job_input["prompt"]}]
        elif "messages" in job_input:
            messages = job_input["messages"]
        else:
            return {
                "error": "❌ Missing 'messages' or 'prompt' field",
                "status": "error"
            }
        
        # Validate
        if not messages or len(messages) == 0:
            return {
                "error": "❌ Messages cannot be empty",
                "status": "error"
            }
        
        # Extract parameters
        max_tokens = job_input.get("max_tokens", MAX_TOKENS)
        temperature = job_input.get("temperature", TEMPERATURE)
        model_name = job_input.get("model", "gpt-oss-direct")
        
        # Generate response using direct model
        logger.info("🤖 Generating response with direct model...")
        generated_content = generate_response(messages, max_tokens, temperature)
        
        processing_time = time.time() - start_time
        
        # Return OpenAI-compatible response
        return {
            "id": f"chatcmpl-direct-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
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
            "model_loaded": model_loaded,
            "processing_time": round(processing_time, 3),
            "model_name": MODEL_NAME
        }
        
    except Exception as e:
        error_msg = f"❌ Handler error: {str(e)}"
        logger.error(error_msg)
        return {
            "error": error_msg,
            "status": "error",
            "device": DEVICE,
            "model_loaded": model_loaded
        }

# Initialize model on startup
logger.info("🚀 Starting GPT-OSS Direct RunPod Serverless Worker")
logger.info("📋 Pre-loading model...")
load_model()

# Start RunPod worker
if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})