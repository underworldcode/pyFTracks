FROM python:3.5-slim

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        bash-completion \
        build-essential \
	gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PYTHONPATH="/opt"
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir  -r requirements.txt
RUN pip install pyFTracks

EXPOSE 8888

