import json


def create_id_to_filename_dict(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)

    images = data.get("images", [])
    id_to_filename = {idx: image["file_name"] for idx, image in enumerate(images)}

    return id_to_filename

