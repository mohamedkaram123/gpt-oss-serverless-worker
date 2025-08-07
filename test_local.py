#!/usr/bin/env python3
"""
Local testing script for GPT-OSS RunPod Serverless Worker
"""

import json
import sys
from handler import handler

def test_messages_format():
    """Test with OpenAI messages format"""
    print("🧪 Testing Messages Format...")
    
    test_input = {
        "input": {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is artificial intelligence? Please provide a brief explanation."}
            ],
            "max_tokens": 150,
            "temperature": 0.7,
            "model": "gpt-oss-120b"
        }
    }
    
    result = handler(test_input)
    print("✅ Messages Format Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

def test_simple_format():
    """Test with simple prompt format"""
    print("\n🧪 Testing Simple Prompt Format...")
    
    test_input = {
        "input": {
            "prompt": "مرحبا! كيف يمكنني استخدام الذكاء الاصطناعي؟",
            "max_tokens": 100,
            "temperature": 0.8
        }
    }
    
    result = handler(test_input)
    print("✅ Simple Format Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

def test_error_handling():
    """Test error handling"""
    print("\n🧪 Testing Error Handling...")
    
    # Test with missing required fields
    test_input = {
        "input": {
            "max_tokens": 100
        }
    }
    
    result = handler(test_input)
    print("✅ Error Handling Result:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result

def test_from_file():
    """Test using test_input.json file"""
    print("\n🧪 Testing from test_input.json...")
    
    try:
        with open('test_input.json', 'r', encoding='utf-8') as f:
            test_input = json.load(f)
        
        result = handler(test_input)
        print("✅ File Input Result:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return result
    except FileNotFoundError:
        print("❌ test_input.json not found")
        return None

def main():
    """Run all tests"""
    print("🚀 Starting GPT-OSS RunPod Serverless Worker Tests\n")
    
    # Test different formats
    test_messages_format()
    test_simple_format()
    test_error_handling()
    test_from_file()
    
    print("\n🎉 All tests completed!")

if __name__ == "__main__":
    main()