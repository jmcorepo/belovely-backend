FROM python:3.12-slim

# ffmpeg is required for the preview cut
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY generate_song.py generate_previews.py keepsake_pdf.py app.py ./

ENV BELOVELY_DATA_DIR=/data
# Hosts inject $PORT; default 8000 locally
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
