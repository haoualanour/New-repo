FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install flask

EXPOSE 80

CMD ["python", "projet_sec_media.py"]
