FROM python:3.11-slim

WORKDIR /app

ENV LSM_BASE=/app

COPY . /app

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir .

EXPOSE 8000

CMD ["uvicorn", "lsm_hand_tracker.main:app", "--host", "0.0.0.0", "--port", "8000"]
