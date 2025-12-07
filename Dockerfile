# ===== Base =====
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Put NLTK data in a shared, readable path
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /app

# System packages
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      openjdk-21-jre-headless \
      build-essential \
      gcc \
      curl \
      libglib2.0-0 libsm6 libxext6 libxrender1 \
 && rm -rf /var/lib/apt/lists/*

# Copy requirements early for layer caching
COPY requirements.txt /app/requirements.txt

# Install deps (no duplicate installs needed)
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt \
 && pip cache purge

# --- Download required NLTK data *into* $NLTK_DATA ---
RUN mkdir -p "$NLTK_DATA" && python - <<'PY'
import nltk, os
target = os.environ.get("NLTK_DATA")
for pkg in ["punkt", "punkt_tab", "wordnet", "omw-1.4", "stopwords"]:
    try:
        nltk.download(pkg, download_dir=target, quiet=True)
        print(f"Downloaded {pkg} to {target}")
    except Exception as e:
        print(f"Failed to download {pkg}: {e}")
PY

# Pre-install spaCy model
RUN python -m spacy download en_core_web_sm

# Copy project
COPY . /app

# Copy .env file into container (only if you really want it baked into image)
COPY .env /app/.env

# Optional: non-root user
RUN useradd -m appuser \
 && chown -R appuser:appuser /app \
 && chown -R appuser:appuser /usr/local/share/nltk_data
USER appuser

# Streamlit defaults
ENV STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

EXPOSE 8501

# Healthcheck: use curl (already installed)
HEALTHCHECK --interval=30s --timeout=5s --retries=5 CMD \
  curl -fsS http://localhost:8501/_stcore/health || exit 1

# Start command
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
