FROM python:3.10-slim as build-image

WORKDIR /usr/local/bin/deployment

RUN apt-get update && apt-get upgrade -y
RUN apt-get install -y curl ca-certificates gnupg
RUN apt-get install -y gcc g++ make libffi-dev git cargo

COPY ./ /tmp/build

RUN  (cd /tmp/build \
     && python3 -m venv py3env-dev \
     && . py3env-dev/bin/activate \
     && python3 -m pip install -U -r requirements_dev.txt \
     && python3 setup.py bdist_wheel)


RUN  export APP_HOME=/usr/local/bin/deployment \
     && (cd $APP_HOME \
         && python3 -m venv py3env \
         && . py3env/bin/activate \
         && python3 -m pip install -U pip \
         && python3 -m pip install -U setuptools \
         && python3 -m pip install -U wheel \
         && python3 -m pip install -U --pre torch torchvision --extra-index-url https://download.pytorch.org/whl/nightly/cu117 \
         && python3 -m pip install -U dataset_format_benchmark --find-links=/tmp/build/dist)


FROM python:3.10-slim

ENV  PYTHONPATH=/usr/local/bin/deployment

WORKDIR /usr/local/bin/deployment

COPY --from=build-image /usr/local/bin/deployment/ ./

RUN  groupadd -r appgroup \
     && useradd -r -G appgroup -d /home/appuser appuser \
     && install -d -o appuser -g appgroup /usr/local/bin/deployment/logs

USER  appuser
CMD ["/usr/local/bin/deployment/py3env/bin/python3", "-m", "dataset_format_benchmark"]
