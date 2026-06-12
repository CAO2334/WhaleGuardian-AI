FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=5000
ENV WHALE_ARTIFACT_DIR=artifacts/final_model_04
ENV WHALE_CONFIDENCE_THRESHOLD=0.5

WORKDIR /app

# opencv-python 运行时需要 libGL / glib 等系统库。
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

EXPOSE 5000

CMD ["python", "whale_web/app.py"]
