FROM python:3.8-slim-buster

ENV ARTIFACTS_HOME='/home/artifacts/'

RUN mkdir -p $ARTIFACTS_HOME
COPY ./artifacts/* $ARTIFACTS_HOME

COPY . /package
WORKDIR /package

RUN pip3 install --upgrade pip
RUN pip install . && rm -rf /package

WORKDIR /app
COPY app.py .

CMD ["python", "app.py"]
