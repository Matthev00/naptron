import json
import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Ścieżki do danych
annotations_file = "data/nuimages/annotations/nuimages_v1.0-train_truck.json"  # Ścieżka do pliku JSON z annotacjami
images_dir = "data/coco/images/car"  # Folder z obrazami

# Wczytaj dane z pliku JSON
with open(annotations_file, "r") as file:
    data = json.load(file)

# Zmapuj obrazy według ich ID
image_mapping = {image["id"]: image["file_name"] for image in data["images"]}

# Grupowanie bboxów według `image_id`
images_annotations = {}
for annotation in data["annotations"]:
    image_id = annotation["image_id"]
    bbox = annotation["bbox"]
    if image_id not in images_annotations:
        images_annotations[image_id] = []
    images_annotations[image_id].append(bbox)

# Rysowanie bboxów na obrazach i wyświetlanie
for image_id, bboxes in images_annotations.items():
    # Pobierz nazwę pliku obrazu
    if image_id not in image_mapping:
        print(f"Nie znaleziono pliku obrazu dla ID: {image_id}")
        continue

    image_file = image_mapping[image_id]
    image_path = os.path.join(images_dir, image_file)
    
    if not os.path.exists(image_path):
        print(f"Brak obrazu: {image_path}")
        continue

    # Wczytaj obraz
    with Image.open(image_path) as img:
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        # Rysuj każdy bbox
        for bbox in bboxes:
            x, y, width, height = bbox
            rect = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='red', facecolor='none')
            ax.add_patch(rect)

        # Dodaj tytuł i usuń osie
        plt.title(f"Image ID: {image_id}")
        plt.axis('off')

        plt.show()