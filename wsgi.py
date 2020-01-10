import logging
from eval import app
import zmq

if __name__ != "__main__":
    gunicorn_logger = logging.getLogger('gunicorn.error')
    app.logger.handlers = gunicorn_logger.handlers
    app.logger.setLevel(gunicorn_logger.level)

def run_webapp(host=None,port=None, **kwargs):

    zmq_client_context = zmq.Context()
    
    #  Socket to talk to server
    # print(" * Connecting to zmq server...")
    app.logger.info("Connecting to zmq server...")
    zmq_client_socket = zmq_client_context.socket(zmq.REQ)
    zmq_client_socket.connect("tcp://%s:5555" % (kwargs["zmqserver_ip"],))        
    app.config['zmq_client_socket'] = zmq_client_socket
    if host and port:
        app.run(host=host, port=port)
    else:
        return app

#if __name__ != "__main__":
#    run_webapp()
    
if __name__ == "__main__":
    run_webapp(host='0.0.0.0', port=13000, zmqserver_ip="localhost")
