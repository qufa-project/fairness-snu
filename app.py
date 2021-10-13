from flask import Flask, jsonify, request, abort
from threading import Thread
from .tasks.dim_reduction import run_reduction

app = Flask(__name__)


@app.before_request
def check_api():
    request.get_json(force=True)
    if 'Authorization' in request.headers:
        token = request.headers['Authorization']
        if token != 'ptech-token':
            abort(401)
    else:
        abort(401)


@app.route("/fairness/dim_reduction", methods=['POST'])
def impute():
    filename = request.json['file_name']
    dim = request.json['dim']
    thread = Thread(target=run_reduction, args=([filename, dim]))
    thread.daemon = True
    thread.start()
    return jsonify(
        {'started': True}
    )
