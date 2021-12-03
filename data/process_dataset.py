import os
import shutil
from pathlib import Path

from pycocotools.coco import COCO

CATEGORIES = [
            "airplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "dining table",
            "dog",
            "horse",
            "motorcycle",
            "person",
            "potted plant",
            "sheep",
            "couch",
            "train",
            "tv"
        ]

MODE = "TRAIN"
MAX_IMAGES_PER_CLASS = 2000

if MODE == "TRAIN":
    INPUT_FOLDER = "./train2017/"
    OUTPUT_FOLDER = "./train2017_custom/"
    ANNOTATIONS_FILE_PATH = "./annotations/instances_train2017.json"
elif MODE == "VAL":
    INPUT_FOLDER = "./val2017/"
    OUTPUT_FOLDER = "./val2017_custom/"
    ANNOTATIONS_FILE_PATH = "./annotations/instances_val2017.json"

Path(OUTPUT_FOLDER).mkdir(parents=True, exist_ok=True)


def get_category_ids(coco, selected_categories = CATEGORIES):
    all_categories = coco.loadCats(coco.getCatIds())

    category_names_to_ids = {}
    for c in all_categories:
        name = c["name"]
        if name in selected_categories:
            category_names_to_ids[name] = c["id"]

    return category_names_to_ids


coco = COCO(ANNOTATIONS_FILE_PATH)

category_ids = get_category_ids(coco)
print(category_ids)

image_ids = set()
for category_id in category_ids.values():
    curr_image_ids = coco.getImgIds(catIds=category_id)
    curr_image_ids = curr_image_ids[:MAX_IMAGES_PER_CLASS]
    image_ids.update(curr_image_ids)

print("Number of total images", len(image_ids))

for image_id in image_ids:
    image = coco.loadImgs(image_id)[0]
    image_fname = image["file_name"]

    input_path = os.path.join(INPUT_FOLDER, image_fname)
    output_path = os.path.join(OUTPUT_FOLDER, image_fname)

    shutil.copy(input_path, output_path)
