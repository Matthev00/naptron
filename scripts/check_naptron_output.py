import pickle

def extract_bounding_boxes(input_file, output_file):
    # Wczytaj dane z pliku wejściowego
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    # Sprawdź strukturę danych i wyodrębnij bounding boxy
    bounding_boxes = []
    for entry in data:
        bboxes = entry[0]  # Zakładamy, że bounding boxy są w pierwszym elemencie
        bounding_boxes.append(bboxes)

    # Zapisz wyodrębnione bounding boxy do nowego pliku
    with open(output_file, 'wb') as f:
        pickle.dump(bounding_boxes, f)

    print(f"Zapisano bounding boxy do pliku: {output_file}")


# Ścieżki do plików
input_file = 'work_dirs/test_results.pkl'  # Plik wejściowy z pełnymi danymi
output_file = 'bounding_boxes.pkl'  # Plik wynikowy z samymi bounding boxami

# Wywołanie funkcji
extract_bounding_boxes(input_file, output_file)