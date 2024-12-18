import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2


def read_bounding_boxes(file_path):
    """
    Wczytaj bounding boxy z pliku.
    :param file_path: Ścieżka do pliku z bounding boxami.
    :return: Lista bounding boxów.
    """
    with open(file_path, 'rb') as f:
        bounding_boxes = pickle.load(f)

    print(f"Wczytano {len(bounding_boxes)} zestawów bounding boxów.")
    return bounding_boxes


def draw_bounding_boxes(image_path, bounding_boxes, class_names=None):
    """
    Wyświetl obraz z narysowanymi bounding boxami.
    :param image_path: Ścieżka do obrazka.
    :param bounding_boxes: Bounding boxy dla obrazka.
    :param class_names: Lista nazw klas (opcjonalnie).
    """
    # Wczytaj obraz
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Utwórz wykres
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(image)

    # Iteruj przez klasy i ich bounding boxy
    for class_idx, bbox_list in enumerate(bounding_boxes):
        for bbox in bbox_list:
            x1, y1, x2, y2, confidence = bbox
            width, height = x2 - x1, y2 - y1

            # Dodaj prostokąt (bounding box)
            rect = patches.Rectangle(
                (x1, y1), width, height, linewidth=2,
                edgecolor='r', facecolor='none'
            )
            ax.add_patch(rect)

            # Dodaj etykietę
            label = f"Class {class_idx}" if class_names is None else class_names[class_idx]
            label += f" ({confidence:.2f})"
            ax.text(x1, y1 - 10, label, color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

    plt.axis('off')
    plt.show()


# Ścieżka do pliku z bounding boxami
bounding_boxes_file = 'bounding_boxes.pkl'

# Ścieżka do obrazka (zamień na właściwą ścieżkę)
image_path = 'data/coco/images/test2017/000000000001.jpg'

# Odczytaj bounding boxy
bounding_boxes = read_bounding_boxes(bounding_boxes_file)

# Narysuj bounding boxy tylko dla jednego obrazka
print("Rysowanie bounding boxów dla jednego obrazka...")
draw_bounding_boxes(image_path, bounding_boxes[0])  # Używamy bounding boxów dla pierwszego obrazka
