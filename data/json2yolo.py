import json
from os import listdir
from os.path import isfile, join
from utils import *


def convert_coco_json(images_path, labels_path, json_file_path, use_segments=False, cls91to80=False):
    # save_dir = make_dirs()  # output directory
    coco80 = coco91_to_coco80_class()
    custom_20_coco_ids, custom_20_custom_ids = coco_custom_20_classes()
    custom_20_custom_ids_inverted = {v: k for k, v in custom_20_custom_ids.items()}
    imgfiles = set([f for f in listdir(images_path) if isfile(join(images_path, f))])

    Path(labels_path).mkdir(parents=True, exist_ok=True)

    # Import json
    with open(json_file_path) as f:
        data = json.load(f)

    # Create image dict
    images = {'%g' % x['id']: x for x in data['images']}

    # Write labels file
    for x in tqdm(data['annotations'], desc=f'Annotations {json_file_path}'):
        if x['iscrowd']:
            continue

        img = images['%g' % x['image_id']]
        h, w, f = img['height'], img['width'], img['file_name']

        if f not in imgfiles:
            # image not in custom dataset; skip
            continue

        # The COCO box format is [top left x, top left y, width, height]
        box = np.array(x['bbox'], dtype=np.float64)
        box[:2] += box[2:] / 2  # xy top-left corner to center
        box[[0, 2]] /= w  # normalize x
        box[[1, 3]] /= h  # normalize y

        # Segments
        if use_segments:
            segments = [j for i in x['segmentation'] for j in i]  # all segments concatenated
            s = (np.array(segments).reshape(-1, 2) / np.array([w, h])).reshape(-1).tolist()

        # Write
        coco_cat_id = coco80[x['category_id']] if cls91to80 else x['category_id']  # class
        if box[2] > 0 and box[3] > 0 and coco_cat_id in custom_20_coco_ids:  # if w > 0 and h > 0
            coco_cat_name = custom_20_coco_ids[coco_cat_id]
            custom_20_custom_id = custom_20_custom_ids_inverted[coco_cat_name] - 1  # 0 <= id <= n-1
            line = custom_20_custom_id, *(s if use_segments else box)  # cls, box or segments
            with open((Path(labels_path) / f).with_suffix('.txt'), 'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')


if __name__ == '__main__':
    images_path = '../coco_custom2_yolo/images/train2017_custom2_yolo'
    labels_path = '../coco_custom2_yolo/labels/train2017_custom2_yolo'
    json_file_path = '../coco_custom2_yolo/annotations/instances_train2017_custom2_yolo.json'
    convert_coco_json(images_path, labels_path, json_file_path)
