from gevent.pywsgi import WSGIServer
from app import app

PORT = 5050

http_server = WSGIServer(('', PORT), app.server)
http_server.serve_forever()
