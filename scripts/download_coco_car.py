import fiftyone.zoo as foz
import os
import shutil
import json


def prepare_naptron_data(destination_path):
    coco_path = os.path.join(destination_path, "coco")
    car_images_path = os.path.join(coco_path, "images", "car")
    car_annotations_path = os.path.join(coco_path, "annotations", "val_car.json")

    os.makedirs(car_images_path, exist_ok=True)
    os.makedirs(os.path.dirname(car_annotations_path), exist_ok=True)

    car_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["car"],
        max_samples=100
    )

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "car"},
        ]
    }

    annotation_id = 1
    for sample in car_dataset:
        if sample.ground_truth is not None:
            vehicles = [d for d in sample.ground_truth.detections if d.label == "car"]
            if vehicles:
                coco_data["images"].append({
                    "id": sample.id,
                    "file_name": os.path.basename(sample.filepath),
                    "width": sample.metadata.width,
                    "height": sample.metadata.height
                })

                shutil.copy(sample.filepath, car_images_path)

                for vehicle in vehicles:
                    category_id = 1
                    x, y, w, h = vehicle.bounding_box
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": sample.id,
                        "category_id": category_id,
                        "bbox": [
                            x * sample.metadata.width,
                            y * sample.metadata.height,
                            w * sample.metadata.width,
                            h * sample.metadata.height,
                        ],
                        "area": w * h * sample.metadata.width * sample.metadata.height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

    with open(car_annotations_path, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"Zapisano {len(coco_data['images'])} obrazów z pojazdami w katalogu: {car_images_path}")
    print(f"Zapisano adnotacje w pliku: {car_annotations_path}")


if __name__ == "__main__":
    prepare_naptron_data("data")
