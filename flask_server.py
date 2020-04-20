from gevent.pywsgi import WSGIServer
from app import app

PORT = 8050

http_server = WSGIServer(('', PORT), app.server)
http_server.serve_forever()
