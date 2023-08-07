FROM python:3.11-slim-buster
RUN apt update -y && apt install awscli -y
WORKDIR /End-to-End-Image-Captioning

COPY . /End-to-End-Image-Captioning
RUN pip install -r requirements.txt

CMD ["python3", "app.py"]