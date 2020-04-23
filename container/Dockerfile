FROM python:3.7

ADD requirements.txt /

RUN pip install -r requirements.txt

ADD *.py /
ADD assets /assets

CMD ["python", "flask_server.py"]
