FROM python:3.5-slim

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        bash-completion \
        build-essential \
	gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip
RUN pip install git+https://github.com/rbeucher/pyFTracks.git --prefix="/opt"
ENV PYTHONPATH="/opt"

EXPOSE 8888

