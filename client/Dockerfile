FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PYTHONPATH=/app
WORKDIR /app
COPY client/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared ./shared
COPY client ./client
EXPOSE 8080
CMD ["gunicorn", "--chdir", "/app/client", "-b", "0.0.0.0:8080", "--timeout", "150", "app:app"]
