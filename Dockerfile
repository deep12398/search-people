FROM python:3.11-slim

# Install Node.js (needed for Claude Code CLI)
RUN apt-get update && apt-get install -y curl && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs && \
    rm -rf /var/lib/apt/lists/*

# Install Claude Code CLI globally
RUN npm install -g @anthropic-ai/claude-code

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY . .

# Railway sets PORT env var
CMD ["sh", "-c", "uvicorn src.web:app --host 0.0.0.0 --port ${PORT:-8080}"]
