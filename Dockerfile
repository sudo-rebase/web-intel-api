FROM python:3.12-slim

WORKDIR /app

# System deps: lxml needs libxml2/libxslt; playwright needs Chromium deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
RUN pip install --no-cache-dir \
    "fastapi>=0.115" \
    "uvicorn[standard]>=0.32" \
    "pydantic>=2.10" \
    "httpx>=0.28" \
    "beautifulsoup4>=4.12" \
    "lxml>=5.3" \
    "trafilatura>=2.0" \
    "python-dateutil>=2.9" \
    "playwright>=1.49" \
    "python-multipart>=0.0.18"

# Install Chromium for JS rendering support
RUN playwright install chromium --with-deps

COPY app.py .

EXPOSE 8000

CMD ["python", "app.py"]
