import os
import json
import numpy


def read_csv(csv):
    #return np.array([[float(j) for j in i.split(",")] for i in csv.splitlines()])
    return csv.splitlines()


def input_handler(data, context):
    """Pre-process request input before it is sent to TensorFlow Serving REST API
    Args:
        data (obj): the request data stream
        context (Context): an object containing request and configuration details
    Returns:
        (dict): a JSON-serializable dict that contains request body and headers
    """

    if context.request_content_type == "text/csv":

        payload = data.read().decode("utf-8")
        inputs = read_csv(payload)

        input = {"inputs": inputs}

        return json.dumps(input)

    raise ValueError(
        '{{"error": "unsupported content type {}"}}'.format(
            context.request_content_type or "unknown"
        )
    )


def output_handler(data, context):
    """Post-process TensorFlow Serving output before it is returned to the client.
    Args:
        data (obj): the TensorFlow serving response
        context (Context): an object containing request and configuration details
    Returns:
        (bytes, string): data to return to client, response content type
    """

    if data.status_code != 200:
        raise ValueError(data.content.decode("utf-8"))

    response_content_type = context.accept_header
    prediction = data.content
    return prediction, response_content_type
                    
                                            
    
    
