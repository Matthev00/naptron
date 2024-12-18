import os
from collections import Counter

directories = ["data/coco/images/ood", "data/coco/images/car"]

def collect_filenames(dirs):
    all_filenames = []
    for dir_path in dirs:
        if os.path.exists(dir_path):
            for file in os.listdir(dir_path):
                if os.path.isfile(os.path.join(dir_path, file)):
                    all_filenames.append(file)
        else:
            print(f"Directory {dir_path} does not exist.")
    return all_filenames

all_filenames = collect_filenames(directories)

filename_counts = Counter(all_filenames)

unique_files = sum(1 for count in filename_counts.values() if count == 1)

repeated_files = sum(1 for count in filename_counts.values() if count > 1)

# Wyniki
print(f"Total files: {len(all_filenames)}")
print(f"Unique files: {unique_files}")
print(f"Repeated files: {repeated_files}")
