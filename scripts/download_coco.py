import fiftyone.zoo as foz
import fiftyone.types as fot
import os
import shutil


def prepare_naptron_data(destination_path):
    coco_path = os.path.join(destination_path, "coco")
    images_val_path = os.path.join(coco_path, "images", "val2017")
    images_test_path = os.path.join(coco_path, "images", "test2017")
    annotations_val_path = os.path.join(coco_path, "annotations", "instances_val2017.json")
    annotations_test_path = os.path.join(coco_path, "annotations", "image_info_test2017.json")

    os.makedirs(images_val_path, exist_ok=True)
    os.makedirs(images_test_path, exist_ok=True)
    os.makedirs(os.path.dirname(annotations_val_path), exist_ok=True)

    val_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="validation",
        label_types=["detections"],
        classes=["car"],
        max_samples=100
    )

    temp_val_export_dir = os.path.join(destination_path, "temp_coco_val_export")
    val_dataset.export(
        export_dir=temp_val_export_dir,
        dataset_type=fot.COCODetectionDataset,
        label_field="ground_truth"
    )

    val_exported_images = os.path.join(temp_val_export_dir, "data")
    if os.path.exists(val_exported_images):
        for file in os.listdir(val_exported_images):
            shutil.move(os.path.join(val_exported_images, file), images_val_path)

    val_exported_annotations = os.path.join(temp_val_export_dir, "labels.json")
    if os.path.exists(val_exported_annotations):
        shutil.move(val_exported_annotations, annotations_val_path)

    test_dataset = foz.load_zoo_dataset(
        "coco-2017",
        split="test",
        label_types="detections",
        classes=["car"],
        max_samples=100
    )

    temp_test_export_dir = os.path.join(destination_path, "temp_coco_test_export")
    test_dataset.export(
        export_dir=temp_test_export_dir,
        dataset_type=fot.COCODetectionDataset
    )

    # Przenoszenie testowych obrazów i anotacji
    test_exported_images = os.path.join(temp_test_export_dir, "data")
    if os.path.exists(test_exported_images):
        for file in os.listdir(test_exported_images):
            shutil.move(os.path.join(test_exported_images, file), images_test_path)

    test_exported_annotations = os.path.join(temp_test_export_dir, "labels.json")
    if os.path.exists(test_exported_annotations):
        shutil.move(test_exported_annotations, annotations_test_path)

    # Usuwanie tymczasowych katalogów eksportu
    shutil.rmtree(temp_val_export_dir)
    shutil.rmtree(temp_test_export_dir)


if __name__ == "__main__":
    prepare_naptron_data("data")
