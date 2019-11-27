FROM python:3.7

RUN pip install Click==7.0 dash==1.6.1 dash-core-components==1.5.1
RUN pip install dash-daq==0.3.1 dash-html-components==1.0.2 dash-renderer==1.2.1
RUN pip install dash-table==4.5.1 Flask==1.1.1 Flask-Compress==1.4.0
RUN pip install future==0.18.2 gevent==1.4.0 greenlet==0.4.15
RUN pip install itsdangerous==1.1.0 Jinja2==2.10.3 MarkupSafe==1.1.1
RUN pip install numpy==1.17.4 pandas==0.25.3 plotly==4.3.0
RUN pip install python-dateutil==2.8.1 pytz==2019.3 retrying==1.3.3
RUN pip install six==1.13.0 Werkzeug==0.16.0

ADD *.py /
ADD assets /assets

CMD ["python", "flask_server.py"]
