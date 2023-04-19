import pandas as pd

df = pd.read_csv('./data/metadata.csv')
# drop the category_id column
df = df.drop(['category_id'], axis=1)
# given an image, only keep one bounding box as our computational resources are limited
df = df.drop_duplicates('img_id', ignore_index=True)
# export filtered metadata
df.to_csv('./data/filtered_metadata.csv', index=False)