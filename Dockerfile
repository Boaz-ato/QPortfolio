FROM python:3.11-slim

RUN apt-get update && apt-get install -y gcc g++ libssl-dev curl
RUN if [ "$(arch)" = aarch64 ]; then apt-get install -y build-essential; fi

# Install Julia (needed for JuliQAOATrainer)
RUN curl -fsSL https://install.julialang.org | sh -s -- --yes
ENV PATH="/root/.juliaup/bin:$PATH"

RUN mkdir /opt/project

COPY requirements.txt requirements.txt
COPY requirements-dev.txt requirements-dev.txt

RUN pip3 install --upgrade pip

RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements-dev.txt

WORKDIR /opt/project
COPY . /opt/project

ENV PYTHONPATH=/opt/project/src
