import fiftyone.zoo as foz
import os
import shutil
import json


def prepare_naptron_data(destination_path):
    coco_path = os.path.join(destination_path, "coco")
    ood_images_path = os.path.join(coco_path, "images", "ood")
    ood_annotations_path = os.path.join(coco_path, "annotations", "ood.json")

    os.makedirs(ood_images_path, exist_ok=True)
    os.makedirs(os.path.dirname(ood_annotations_path), exist_ok=True)

    # Pobieranie samochodów z danych testowych
    ood_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["truck", "bus"],
        max_samples=100
    )

    coco_data = {
        "images": [],
        "annotations": [],
        "categories": [
            {"id": 1, "name": "truck"},
            {"id": 2, "name": "bus"}
        ]
    }

    annotation_id = 1
    for sample in ood_dataset:
        if sample.ground_truth is not None:
            vehicles = [d for d in sample.ground_truth.detections if d.label in ["truck", "bus"]]
            if vehicles:
                coco_data["images"].append({
                    "id": sample.id,
                    "file_name": os.path.basename(sample.filepath),
                    "width": sample.metadata.width,
                    "height": sample.metadata.height
                })

                shutil.copy(sample.filepath, ood_images_path)

                for vehicle in vehicles:
                    category_id = 1 if vehicle.label == "truck" else 2
                    x, y, w, h = vehicle.bounding_box
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": sample.id,
                        "category_id": category_id,
                        "bbox": [
                            x * sample.metadata.width,  # Zamiana na współrzędne pikselowe
                            y * sample.metadata.height,
                            w * sample.metadata.width,
                            h * sample.metadata.height,
                        ],
                        "area": w * h * sample.metadata.width * sample.metadata.height,
                        "iscrowd": 0
                    })
                    annotation_id += 1

    with open(ood_annotations_path, "w") as f:
        json.dump(coco_data, f, indent=4)

    print(f"Zapisano {len(coco_data['images'])} obrazów z pojazdami w katalogu: {ood_images_path}")
    print(f"Zapisano adnotacje w pliku: {ood_annotations_path}")



if __name__ == "__main__":
    prepare_naptron_data("data")
