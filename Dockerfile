FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY modelwatch ./modelwatch
RUN python -m modelwatch.train --output /app/artifacts
RUN useradd --system --uid 10001 worker && chown -R worker /app
USER worker
EXPOSE 8000
CMD ["uvicorn", "modelwatch.api:app", "--host", "0.0.0.0", "--port", "8000"]
