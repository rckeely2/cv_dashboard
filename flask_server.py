from gevent.pywsgi import WSGIServer
from app import app

http_server = WSGIServer(('', 5000), app.server)
http_server.serve_forever()
