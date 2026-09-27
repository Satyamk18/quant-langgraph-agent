# ==============================================================================
# AlphaAgent Production Container
# Enterprise deployment container for CLI, Agent Workflows, and MCP Server
# ==============================================================================

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies (build-essential for C extensions if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python requirements first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code and filing datasets
COPY src/ ./src/
COPY data/ ./data/
COPY evals/ ./evals/
COPY main.py .

# Create outputs directory
RUN mkdir -p outputs data/chroma_db

# Default to running the AlphaAgent CLI (can be overridden to run src/mcp_server.py)
CMD ["python", "main.py"]
