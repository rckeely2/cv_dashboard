# minimal_dash_flask
Minimal test for Dash in a Flask

Clone repository
``` shell
git clone https://github.com/rckeely/minimal_dash_flask.git
```

Enter folder
``` shell
cd minimal_dash_flask/
```

Generate virtual environment
``` shell
python3 -m venv venv && source venv/bin/activate
```

I installed the following with pip:
``` shell
pip install dash==1.6.1 dash-daq==0.3.1 pandas gevent
```

And then saved this to requirements.txt
``` shell
pip freeze -l > requirements.txt
```

So it should be possible to just regenerate with
``` shell
pip install -r requirements.txt
```

``` shell
docker build -t myname/streamcont .
```

Then I made the files and folders:
``` shell
touch app.py
touch flask_server.py
touch
mkdir assets
touch assets/main.css
mkdir container
touch container/Dockerfile
touch deploy.sh
```

Then I added the following content to `app.py`
``` python
import dash
import dash_html_components as html
import dash_core_components as dcc

# def create_layout():
#     return

app = dash.Dash(__name__)
app.layout = html.Div(className="container",
                                children=["Hello World"])

if __name__ == "__main__":
    app.run_server(port=8050, debug=True)
```

Test ```app.py``` in isolation by running, then navigating to [ http://localhost:8050 ]:
``` shell
python app.py
```


To `flask_server.py`
``` python
from gevent.pywsgi import WSGIServer
from app import app

PORT = 8050

http_server = WSGIServer(('', PORT), app.server)
http_server.serve_forever()
```

To `deploy.sh`
```
#!/bin/sh

cp *.py container
cp -r assets container/assets

```

To `container/Dockerfile`
```
FROM python:3.7

RUN pip install  dash==1.6.1 dash-daq==0.3.1 pandas gevent

ADD *.py /
ADD assets /assets

CMD ["python", "flask_server.py"]

```

And built the container:
``` shell
cd container
docker build -t rckeely/min_dash_flask .
```

Test the container locally by running the container and the navigating to
[ http://localhost:8050 ]:
```
docker run -d --name mdf_app -p 8050:8050 rckeely/min_dash_flask
```

Login to docker
``` shell
docker login
```

Then I tag the release and push to docker
``` shell
docker push rckeely/min_dash_flash
```

Check if anything else is listening on the port
``` shell
sudo netstat -tulpn | grep LISTEN
```

Run the server on 80 (stop anything else running on the same port first)
```
docker run -d --name mdf_app -p 80:8050 rckeely/min_dash_flask
```

Navigate to the server [ http://ec2-52-207-180-159.compute-1.amazonaws.com/ ]
