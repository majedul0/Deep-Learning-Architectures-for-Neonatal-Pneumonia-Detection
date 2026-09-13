# Hugging Face Spaces (Docker SDK) deployment for the pneumonia classifier dashboard.
# See webapp/README.md for the full setup steps (upload weights, create the Space, push).

FROM python:3.12-slim

WORKDIR /app

# opencv-python-headless still needs these for image codec support on a slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt

COPY . .

# Bake the trained model weights into the image at build time, fetched from the
# public Hugging Face Hub model repo configured in webapp/hf_config.py -- this
# keeps the running container self-contained (no runtime network dependency).
RUN python webapp/download_models.py

ENV PORT=7860
EXPOSE 7860

# Single worker: each gunicorn worker keeps its own in-memory model cache, and
# duplicating that across workers would multiply RAM use for little benefit on
# a low-traffic demo. Threads handle concurrent requests within that worker.
CMD ["sh", "-c", "gunicorn --chdir webapp --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 app:app"]
