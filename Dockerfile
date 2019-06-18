FROM python:3.5-slim

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        bash-completion \
        build-essential \
	python3.5-dbg \
	strace \
	gdb \
	gcc && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN mkdir pyTracks

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir  -r requirements.txt
#RUN cat "source /usr/share/gdb/auto-load/usr/bin/python3.5-gdb.py" >> .gdbinit 

EXPOSE 8888

WORKDIR pyTracks
