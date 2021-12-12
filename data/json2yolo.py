import json
from os import listdir
from os.path import isfile, join
from utils import *


def convert_coco_json(images_path, json_file_path, use_segments=False, cls91to80=False):
    # save_dir = make_dirs()  # output directory
    coco80 = coco91_to_coco80_class()
    imgfiles = set([f for f in listdir(images_path) if isfile(join(images_path, f))])

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
        if box[2] > 0 and box[3] > 0:  # if w > 0 and h > 0
            cls = coco80[x['category_id'] - 1] if cls91to80 else x['category_id'] - 1  # class
            line = cls, *(s if use_segments else box)  # cls, box or segments
            with open((Path(images_path) / f).with_suffix('.txt'), 'a') as file:
                file.write(('%g ' * len(line)).rstrip() % line + '\n')


if __name__ == '__main__':
    images_path = '../coco-custom/images/val2017_custom_2'
    json_file_path = '../coco-custom/annotations/instances_val2017_custom_2.json'
    convert_coco_json(images_path, json_file_path)
