FROM python:3.6

RUN mkdir /code
WORKDIR /code

ADD ./subscriber.py /code/
ADD ./constants.py /code/
ADD ./util.py /code/
ADD ./feature_ingress.py /code/
ADD ./requirements.txt /code/
ADD ./sensor_key.json /code/

RUN pip install -r requirements.txt