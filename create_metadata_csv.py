import json

import pandas as pd


def preprocess(annotations):
    # preprocesses a list of dictionaries w/ the keys
    # bbox
    # image_id
    # height
    # width
    # category_id
    # mask
    # expressions

    # step 1: delete the masks from each annotation
    for annotation in annotations:
        del annotation['mask']
    processed_annotations = []
    duplicates = set()
    # step 2: remove duplicate annotations
    for annotation in annotations:
        annotation_str = str(annotation)
        if annotation_str not in duplicates:
            duplicates.add(annotation_str)
            processed_annotations.append(annotation)

    return processed_annotations


# various needed paths
refcoco = './data/refcoco-unc/instances.json'
refcocoplus = './data/refcocoplus-unc/instances.json'
refcocog = './data/refcocog-umd/instances.json'
images = './data/images'

# initialize dataframe
cols = ['img_id', 'expression', 'img_width', 'img_height', 'category_id', 'cx', 'cy', 'w', 'h']
df = pd.DataFrame(columns=cols)

# loop over jsons and add info to the dataframe
img_ids = set()
count = 0
for json_path in [refcoco, refcocoplus, refcocog]:
    data = json.load(open(json_path, 'r'))
    for annotations in data.values():
        annotations = preprocess(annotations)
        for annotation in annotations:
            img_id = '{}/{}.jpg'.format(images, annotation['image_id'])
            expression = max(annotation['expressions'], key=len)  # just use the longest expression for each image
            img_width = annotation['width']
            img_height = annotation['height']
            category_id = annotation['category_id']
            cx, cy, w, h = annotation['bbox']
            values = [img_id, expression, img_width, img_height, category_id, cx, cy, w, h]
            row = pd.DataFrame([dict(zip(cols, values))])
            df = pd.concat([df, row], ignore_index=True)
            img_ids.add(img_id)
            print(count, img_id, len(img_ids))
            count += 1

df.to_csv('./data/metadata.csv', index=False)
