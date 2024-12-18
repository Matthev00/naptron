import os
import mmcv
import cv2
import matplotlib.pyplot as plt
import json
import torch

# Ścieżki do plików
#json_file = "data/nuimages/annotations/nuimages_v1.0-train_car_filtered.json"  # Ścieżka do pliku JSON
json_file = "data/nuimages/annotations/nuimages_v1.0-train_truck_small.json"  # Ścieżka do pliku JSON
json_file = "data/nuimages/annotations/nuimages_v1.0-train_car_filtered_small.json"
output_folder = "visualizations_small/"  # Folder, gdzie zapiszemy wizualizacje
base_image_path = "data/nuimages/samples/"  # Folder z obrazami

# 1 STEP
pkl_file = "work_dirs/outputs_dump/faster_rcnn_naptron_voc2coco_model_outputs.pkl"  # Plik z wynikami modelu

# 2 STEP
# pkl_file = "work_dirs/outputs_dump/faster_rcnn_r50_fpn_voc0712_cocofmt_naptron_trainset_model_outputs.pkl"  # Plik z wynikami modelu

# Upewnij się, że folder wyników istnieje
os.makedirs(output_folder, exist_ok=True)


# Funkcja do parsowania danych z pliku PKL
def parse_nap_detector_outputs(outputs):
    bboxes, activations, shapes, labels = outputs[::4], outputs[1::4], outputs[2::4], outputs[3::4]
    if not isinstance(activations[0], torch.Tensor):
        activations = [i[0] for i in activations]
    if len(bboxes[0]) == 1:
        bboxes = [i[0] for i in bboxes]
    labels = [i.cpu() if isinstance(i, torch.Tensor) else i[0].cpu() for i in labels]
    return bboxes, activations, labels


# Wczytaj dane z plików
results = mmcv.load(pkl_file)
with open(json_file, "r") as f:
    json_data = json.load(f)

# Wyodrębnij bboxy, aktywacje i etykiety z PKL
bboxes, _, labels = parse_nap_detector_outputs(results)

# Mapowanie zdjęć po ich ID z JSON
image_data = {image["id"]: image for image in json_data["images"]}

# Iteracja przez dane detekcji z PKL
for img_id, img_bboxes, img_labels in zip(range(1948, 1968), bboxes, labels):
    if img_id not in image_data:
        print(f"Brak danych dla image_id {img_id}")
        continue

    # Pobierz informacje o obrazie z JSON
    image_info = image_data[img_id]
    image_path = os.path.join(base_image_path, image_info["file_name"].split("samples/")[-1])
    image = cv2.imread(image_path)

    if image is None:
        print(f"Nie znaleziono obrazu: {image_path}")
        continue

    # Narysuj wszystkie bboxy na obrazie
    for bbox, label in zip(img_bboxes, img_labels):
        if bbox.size == 0:
            continue
        for box in bbox:
            x1, y1, x2, y2, score = int(box[0]), int(box[1]), int(box[2]), int(box[3]), box[4],  # Opcjonalny próg pewności
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"Class: {label}, Score: {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Zapisz obraz z narysowanymi bboxami
    output_path = os.path.join(output_folder, f"result_{img_id}.jpg")
    cv2.imwrite(output_path, image)

    # Opcjonalnie wyświetl obraz
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(f"Image ID: {img_id}")
    plt.axis("off")
    plt.show()

print(f"Wizualizacje zapisane w: {output_folder}")
