FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY models /app/models  # Assume models copied

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "80"]