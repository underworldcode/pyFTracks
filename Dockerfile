FROM python:3.7-slim

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        bash-completion \
        build-essential \
	gcc \
	git && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /opt
RUN git clone https://github.com/rbeucher/pyFTracks.git
RUN pip install numpy
RUN pip install pyFTracks/
ENV PYTHONPATH="/opt"
WORKDIR /home

EXPOSE 8888

