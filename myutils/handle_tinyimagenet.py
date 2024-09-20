# tiny-imagenet-200/
#     ├── train/
#     │   ├── n01443537/
#     │   │   ├── images/
#     │   │   │   ├── n01443537_0.JPEG
#     │   │   │   ├── n01443537_1.JPEG
#     │   │   │   └── ...
#     │   │   └── ...
#     │   ├── n01558993/
#     │   └── ...
#     ├── val/
#     │   ├── images/
#     │   │   ├── val_0.JPEG
#     │   │   ├── val_1.JPEG
#     │   │   └── ...
#     │   └── val_annotations.txt
#     ├── test/
#     │   ├── images/
#     │   │   ├── test_0.JPEG
#     │   │   ├── test_1.JPEG
#     │   │   └── ...
#     └── wnids.txt
import os
import shutil
#把下载的数据改成ImageFolder能识别的结构
# root_directory/
#     class1/
#         image1.jpg
#         image2.jpg
#        ...
#     class2/
#         image3.jpg
#         image4.jpg
#        ...
#    ...
# Paths
val_dir = 'tiny-imagenet-200/val'
val_img_dir = os.path.join(val_dir, 'images')
val_annotations = os.path.join(val_dir, 'val_annotations.txt')

# Read val_annotations.txt
with open(val_annotations, 'r') as f:
    annotations = [line.strip().split('\t') for line in f]

# Create directories for each class
for ann in annotations:
    img_name, class_id = ann[0], ann[1]
    class_dir = os.path.join(val_dir, class_id)

    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    # Move images to their respective class folders
    src_img_path = os.path.join(val_img_dir, img_name)
    dst_img_path = os.path.join(class_dir, img_name)
    shutil.move(src_img_path, dst_img_path)

# Remove original 'images/' folder
shutil.rmtree(val_img_dir)
