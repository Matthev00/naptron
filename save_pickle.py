import pickle
import json

output_file = "work_dirs/outputs_dump/faster_rcnn_naptron_voc2coco_model_outputs.pkl"

with open(output_file, "rb") as f:
    data = pickle.load(f)

result = []

for i, image_data in enumerate(data):
    image_result = {
        "image_index": i,
        "bboxes": [],
        "uncertainty_scores": []
    }
    for image in image_data:
        for row in image:
            bbox = row[:4].tolist()
            score = row[4]
            image_result["bboxes"].append(bbox)
            image_result["uncertainty_scores"].append(score)
    result.append(image_result)

output_json = "output_results.json"
with open(output_json, "w") as f:
    json.dump(result, f, indent=4)

print(f"Dane zostały zapisane w pliku: {output_json}")
