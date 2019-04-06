import base64
import os
from functools import wraps

import jwt
from bo.network_dao import get_network
from flask import abort
from flask import request


def check_api_key(request_input):
    '''Check JWT token validity'''
    auth_token = request_input.headers.get('Authorization')
    if auth_token:
        try:
            network_id = os.environ['NETWORK_ID']
            network = get_network(network_id)
            decoded_token = jwt.decode(auth_token, base64.b64decode(network.api_key_secret), algorithms=['HS256'],
                                       options={'verify_aud': False, 'require_sub': True})
            assert decoded_token['name'] == network.name
            assert decoded_token['sub'] == network.user.login + '-' + network.name
            assert decoded_token['scope'] == 'run'

            return True
        except:
            return False


def require_appkey(view_function):
    # '''Wraps functionality to annotation used in controller'''
    @wraps(view_function)
    def decorated_function(*args, **kwargs):
        api_key_result = check_api_key(request)
        if api_key_result:
            return view_function(*args, **kwargs)

        print('request abbort -- 401')
        abort(401)

    return decorated_function
