FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
COPY . /app/

# Install ffmpeg and other necessary packages
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install -r requirements.txt

# https://github.com/pytube/pytube/issues/1498#issuecomment-1475993725
RUN sed -i 's/transform_plan_raw =.*/transform_plan_raw = js/g' /usr/local/lib/python3.10/site-packages/pytube/cipher.py

CMD ["streamlit", "run", "app.py", "--server.port", "8080"]
