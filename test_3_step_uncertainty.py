import json
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from images_dir import create_id_to_filename_dict

image_range = 100

# with open("output_results.json") as f:
#     data = json.load(f)[:image_range]

with open("data/nuimages/annotations/nuimages_v1.0-train_truck.json") as f:
    data = json.load(f)

json_file_path = "data/nuimages/annotations/nuimages_v1.0-train_truck.json"
images_dict = create_id_to_filename_dict(json_file_path)

for image in range(image_range):
    image_id = image + 1
    bboxes = data[image]["bboxes"]
    uncertainty_scores = data[image]["uncertainty_scores"]
    image_path = images_dict[image_id]

    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    for bbox, score in zip(bboxes, uncertainty_scores):
        # if score < 0.5:
        #     continue
        x_min, y_min, x_max, y_max = bbox
        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)
        draw.text((x_min, y_min - 10), f"Uncertainty: {score}", fill="red")

    plt.figure(figsize=(10, 10))
    plt.title(f"Image {image_id}")
    plt.imshow(image)
    plt.axis("off")
    plt.show()

