FROM python:3.7-slim

RUN apt-get update -qq && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq --no-install-recommends \
        bash-completion \
        build-essential \
	gcc \
	git \
	tk8.6 && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip

WORKDIR /opt
RUN git clone https://github.com/rbeucher/pyFTracks.git
RUN pip install -e pyFTracks/

RUN useradd -ms /bin/bash jovyan
USER jovyan

WORKDIR /home/jovyan
RUN cp -rf /opt/pyFTracks/docs .

ENV DISPLAY :0
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip='0.0.0.0'", "--no-browser"]
