# GPT-OSS 120B RunPod Serverless Worker

A RunPod Serverless worker for GPT-OSS 120B model, built following RunPod's official template structure for maximum reliability and compatibility.

## 🚀 Features

- **RunPod Serverless Compatible**: Built using official RunPod worker template
- **GPT-OSS 120B Support**: Optimized for GPT-OSS 120B model
- **Dual Input Format**: Supports both simple prompt and OpenAI messages format
- **Mock Mode**: Works without GPT-OSS API for testing
- **Error Handling**: Comprehensive error handling and logging
- **Cost Effective**: Pay only for actual usage

## 📁 Project Structure

```
gpt-oss-serverless-worker/
├── handler.py           # Main serverless handler
├── Dockerfile          # RunPod base image configuration
├── requirements.txt    # Python dependencies
├── test_input.json     # Test input (messages format)
├── test_simple.json    # Test input (simple format)
└── README.md          # This file
```

## 🛠️ Local Testing

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Test with Messages Format

```bash
python handler.py
```

This will use `test_input.json` by default.

### 3. Test with Simple Format

```bash
# Modify handler.py to load test_simple.json or test directly:
python -c "
from handler import handler
import json

with open('test_simple.json', 'r') as f:
    test_input = json.load(f)

result = handler(test_input)
print(json.dumps(result, indent=2))
"
```

## 🌐 Deployment to RunPod Serverless

### 1. Build and Push Docker Image

```bash
# Build the image
docker build -t your-username/gpt-oss-serverless-worker:latest .

# Push to Docker Hub
docker push your-username/gpt-oss-serverless-worker:latest
```

### 2. Create RunPod Serverless Endpoint

1. Go to [RunPod Console](https://www.runpod.io/console/serverless)
2. Click **"Create Endpoint"**
3. Configure:
   - **Name**: `gpt-oss-120b-worker`
   - **Container Image**: `your-username/gpt-oss-serverless-worker:latest`
   - **Container Disk**: `5GB`
   - **GPU**: `A100 80GB` or `H100` (for 120B model)
   - **Max Workers**: `1-3`
   - **Idle Timeout**: `30` seconds

### 3. Environment Variables

Set these in RunPod Console:

```
GPT_OSS_API_URL=https://your-gpt-oss-endpoint.com
GPT_OSS_API_KEY=your_api_key_here
DEFAULT_MAX_TOKENS=2048
DEFAULT_TEMPERATURE=0.7
```

## 📝 API Usage

### Input Format 1: Messages (OpenAI Compatible)

```json
{
  "input": {
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "What is artificial intelligence?"}
    ],
    "max_tokens": 150,
    "temperature": 0.7,
    "model": "gpt-oss-120b"
  }
}
```

### Input Format 2: Simple Prompt

```json
{
  "input": {
    "prompt": "Hello World",
    "max_tokens": 100,
    "temperature": 0.7
  }
}
```

### Example API Call

```bash
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Authorization: Bearer YOUR_RUNPOD_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "messages": [
        {"role": "user", "content": "مرحبا! كيف حالك؟"}
      ],
      "max_tokens": 100
    }
  }'
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GPT_OSS_API_URL` | GPT-OSS API endpoint | `http://localhost:8000` |
| `GPT_OSS_API_KEY` | API key for GPT-OSS | `""` |
| `DEFAULT_MAX_TOKENS` | Default max tokens | `2048` |
| `DEFAULT_TEMPERATURE` | Default temperature | `0.7` |

### Mock Mode

If `GPT_OSS_API_URL` is not configured or set to localhost, the worker will return mock responses for testing purposes.

## 💰 Cost Estimation

### RunPod Serverless Pricing (A100 80GB)

- **Cold Start**: ~$0.02 (45 seconds)
- **Warm Request**: ~$0.0023 (5 seconds)
- **Monthly Cost** (1000 requests/day): ~$69

### Comparison with Other Services

| Service | Cost per Request | Monthly (30k requests) |
|---------|------------------|------------------------|
| **GPT-OSS on RunPod** | $0.0023 | $69 |
| **GPT-4o API** | $0.0050 | $150 |
| **Savings** | **54%** | **$81** |

## 🐛 Troubleshooting

### Common Issues

1. **Handler not starting**:
   - Check Dockerfile syntax
   - Verify requirements.txt dependencies
   - Check RunPod logs

2. **API connection errors**:
   - Verify `GPT_OSS_API_URL` is correct
   - Check `GPT_OSS_API_KEY` if required
   - Test with mock mode first

3. **Timeout errors**:
   - Increase timeout in httpx client
   - Check GPU memory allocation
   - Optimize model loading

### Debug Mode

Enable debug logging by setting:

```python
logging.basicConfig(level=logging.DEBUG)
```

## 📊 Performance Tips

1. **Model Loading**: Load models globally, not in handler function
2. **Connection Pooling**: Reuse HTTP client connections
3. **Batch Processing**: Process multiple requests together when possible
4. **GPU Optimization**: Use appropriate GPU tier for your model size

## 🔒 Security

- Store API keys in environment variables
- Use HTTPS endpoints only
- Validate all input data
- Implement rate limiting if needed

## 📄 License

MIT License - see LICENSE file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📞 Support

- Check RunPod documentation: [docs.runpod.io](https://docs.runpod.io)
- Review logs in RunPod Console
- Test locally before deploying# gpt-oss-serverless-worker
