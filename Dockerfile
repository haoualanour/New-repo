FROM ubuntu:latest

ADD projet_sec_media.py /app/projet_sec_media.py

CMD ["python3", "/app/projet_sec_media.py"]


