FROM python:3.10-slim

WORKDIR /app

COPY "projet sec media.py" .

CMD ["python", "projet sec media.py"]
