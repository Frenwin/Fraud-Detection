FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt
COPY pyproject.toml README.md ./
COPY configs ./configs
COPY sql ./sql
COPY src ./src
RUN pip install --no-cache-dir --no-deps -e .
ENV PYTHONUNBUFFERED=1
EXPOSE 8010
