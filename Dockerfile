FROM python:3.11-slim

WORKDIR /app

# chromadb requires a C compiler for some native extensions
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 7331

# Bind Flask to all interfaces so the published port is reachable from the host
# and from other devices on the same network.
# Override with FLASK_HOST=127.0.0.1 to restrict to localhost only.
ENV FLASK_HOST=0.0.0.0
ENV FLASK_PORT=7331

# Ollama runs on the host machine (GPU/CPU stays there).
# host.docker.internal resolves to the host IP via extra_hosts in docker-compose.
# Override to point at any accessible Ollama instance.
ENV OLLAMA_HOST=http://host.docker.internal:11434

CMD ["python", "main.py"]
