import datetime
import io
import json
import os
import time

import requests
from PIL import Image
from flask import abort
from flask import jsonify, request
from flask_restful_swagger import swagger
from werkzeug.datastructures import FileStorage
from flask_restplus import Resource
from api_security import require_appkey
from bo.network_dao import get_network
from configuration.app_config import app, ns, api, api_port, debug_server
from network_model import load_trained_model, predict




@ns.route('/swagger2.json')
class ApiDocsController(Resource):
    '''Return swagger docs json'''

    def get(self):
        with app.app_context():
            schema = api.__schema__
            schema['basePath'] = os.environ['API_URI']
            return jsonify(schema)


@ns.route('/healtz')
class HealthController(Resource):
    '''Check if app is running and know its id'''

    @swagger.operation()
    def get(self):
        network = get_network(network_id)
        if network is not None:
            return jsonify("true")


@ns.route('/network/predict')
class NetworkController(Resource):
    '''Classify based on user input'''

    post_parser = api.parser()
    post_parser.add_argument('Authorization', type=str, location='headers', required=True)
    post_parser.add_argument('data', type=FileStorage, location='files')

    @api.expect(post_parser, validate=True)
    @require_appkey
    def post(self):
        data = {"success": False}

        input_data, additional_data = self.get_request_data(request)
        if input_data is not None:
            data = predict(input_data, additional_data)
            ts = time.time()
            data['timestamp'] = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
            data["success"] = True
            data["main_class"] = data['predictions'][0]
            main_label = data["main_class"]

            network = get_network(network_id)
            data = self.call_modules(network, main_label, data, requests.Session())
        else:
            abort(400)

        return jsonify(data)

    def get_request_data(self, request):
        if request.files.get("data"):
            image = request.files["data"].read()
            image = Image.open(io.BytesIO(image))
            return image, None
        return None, None

    def call_modules(self, network, main_label, data, session):
        for module in network.modules:
            if module.response_class == main_label and module.status == 4:
                headers = {"Authorization": module.api_key, 'Content-Type': 'application/json'}
                module_data = {}
                module_data['context_data'] = data
                module_data['original_request'] = data['user_request']

                response = session.post('http://' + module.connection_uri_internal, json=module_data, headers=headers)
                data['module_response'] = json.loads(response.content.decode("utf-8"))
        return data


def configure_network():
    global network_id
    network_id = os.environ['NETWORK_ID']
    network = get_network(network_id)
    load_trained_model(network)


if __name__ == "__main__":
    configure_network()
    app.run(debug=debug_server, host="0.0.0.0", port=api_port)

# docker build -t images-pretrained:latest .
# docker tag images-pretrained:latest localhost:5000/images-pretrained:latest
# docker push localhost:5000/images-pretrained:latest


# curl -X POST -H "Authorization: eyJuZXR3b3JrX2lkIjoxLCJhcGlfa2V5IjoiUTB4UloxTlFiR1pSVG1VMmNFOHdjVzVtUmpCYWVHRjVkak5QTUdwd1Eybz0ifQ==" -H "Secret: eyJzZWNyZXQiOiJkR0ZxYm1WSVpYTnNidz09In0=" -F input=@plain.jpg 'localhost/stejskys-dd/network/predict'


# curl -X POST "localhost:5001/network/predict"  -H "accept: application/json" -H "Authorization: eyJuZXR3b3JrX2lkIjoxMzEsImFwaV9rZXkiOiJlbk5OZEdKUmVIWjVjbGhRTmtGS1ZraEZRemRPUzJkNGFsRTNORTh4UW00PSJ9" -H "Secret: eyJzZWNyZXQiOiJkR0ZxYm1WSVpYTnNidz09In0=" -F input=@dog.jpg
# curl -X POST "localhost/stejskys-images-pretrained/network/predict"  -H "accept: application/json" -H "Authorization: eyJuZXR3b3JrX2lkIjoxMzEsImFwaV9rZXkiOiJlbk5OZEdKUmVIWjVjbGhRTmtGS1ZraEZRemRPUzJkNGFsRTNORTh4UW00PSJ9" -H "Secret: eyJzZWNyZXQiOiJkR0ZxYm1WSVpYTnNidz09In0=" -F input=@dog.jpg
