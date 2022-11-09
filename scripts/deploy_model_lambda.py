

"""
This Lambda function deploys the model to SageMaker Endpoint. 
If Endpoint exists, then Endpoint will be updated with new Endpoint Config.
"""

import json
import boto3
import time


sm_client = boto3.client("sagemaker")


def lambda_handler(event, context):

    print(f"Received Event: {event}")

    current_time = time.strftime("%m-%d-%H-%M-%S", time.localtime())
    endpoint_instance_type = event["endpoint_instance_type"]
    model_name = event["model_name"]
    endpoint_config_name = "{}-{}".format(event["endpoint_config_name"], current_time)
    endpoint_name = event["endpoint_name"]

    # Create Endpoint Configuration
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": endpoint_instance_type,
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )
    print(f"create_endpoint_config_response: {create_endpoint_config_response}")

    # Check if an endpoint exists. If no - Create new endpoint, if yes - Update existing endpoint
    list_endpoints_response = sm_client.list_endpoints(
        SortBy="CreationTime",
        SortOrder="Descending",
        NameContains=endpoint_name,
    )
    print(f"list_endpoints_response: {list_endpoints_response}")

    if len(list_endpoints_response["Endpoints"]) > 0:
        print("Updating Endpoint with new Endpoint Configuration")
        update_endpoint_response = sm_client.update_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        print(f"update_endpoint_response: {update_endpoint_response}")
    else:
        print("Creating Endpoint")
        create_endpoint_response = sm_client.create_endpoint(
            EndpointName=endpoint_name, EndpointConfigName=endpoint_config_name
        )
        print(f"create_endpoint_response: {create_endpoint_response}")

    return {"statusCode": 200, "body": json.dumps("Endpoint Created Successfully")}
