FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MPLBACKEND=Agg PYTHONPATH=/app
WORKDIR /app
COPY server/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY shared ./shared
COPY server ./server
EXPOSE 5000
CMD ["gunicorn", "--chdir", "/app/server", "-b", "0.0.0.0:5000", "--timeout", "120", "app:app"]
