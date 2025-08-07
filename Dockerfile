FROM runpod/base:0.6.3-cuda11.8.0

# Set python3.11 as the default python
RUN ln -sf $(which python3.11) /usr/local/bin/python && \
    ln -sf $(which python3.11) /usr/local/bin/python3

# Install dependencies
COPY requirements_direct.txt /requirements_direct.txt
RUN uv pip install --upgrade -r /requirements_direct.txt --no-cache-dir --system

# Add files
ADD handler_direct.py .
ADD test_input.json .

# Run the handler
CMD python -u /handler_direct.py