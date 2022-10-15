import json


def json_store(file_name: str, data: dict):
    with open(file_name, 'w') as fp:
        json.dump(data, fp)


def json_load(file_name: str):
    with open(file_name, 'r') as fp:
        return json.load(fp)

