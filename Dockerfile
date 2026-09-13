# Local Docker run for the pneumonia classifier dashboard.
# Model weights are NOT baked into the image (kept the build fast) -- mount
# your local Results-20260913T045822Z-1-001/Results folder into the container
# at runtime instead. See docker-compose.yml, or run directly with:
#   docker run -p 8000:7860 -v "$(pwd)/Results-20260913T045822Z-1-001/Results:/app/Results-20260913T045822Z-1-001/Results:ro" <image>

FROM python:3.12-slim

WORKDIR /app

# opencv-python-headless still needs these for image codec support on a slim base image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY webapp/requirements.txt webapp/requirements.txt
RUN pip install --no-cache-dir -r webapp/requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860

# Single worker: each gunicorn worker keeps its own in-memory model cache, and
# duplicating that across workers would multiply RAM use for little benefit on
# a low-traffic demo. Threads handle concurrent requests within that worker.
CMD ["sh", "-c", "gunicorn --chdir webapp --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 app:app"]
