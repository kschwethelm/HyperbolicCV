import os
from glob import glob
import shutil

def structure_validation_images(val_dir):
    """ Creates separate folders for each class in val_annotations.txt. """
    img_dir = os.path.join(val_dir, 'images')
    val_txt = os.path.join(val_dir, 'val_annotations.txt')

    if not os.path.exists(val_txt):
        raise RuntimeError(f"Path to validation annotations not found: {val_txt}")

    fp = open(val_txt, 'r')
    data = fp.readlines()

    val_img_dict = {}
    for line in data:
        words = line.split('\t')
        val_img_dict[words[0]] = words[1]
    fp.close()

    # Create subfolders for validation images based on label,
    # and move images into the respective folders
    for img, folder in val_img_dict.items():
        newpath = (os.path.join(img_dir, folder))
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        if os.path.exists(os.path.join(img_dir, img)):
            os.rename(os.path.join(img_dir, img), os.path.join(newpath, img))

def structure_training_images(train_dir):
    """ Creates a parent image folder. """
    img_dir = os.path.join(train_dir, 'images')

    class_dirs = glob(train_dir+"/*")

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)

    for dir in class_dirs:
        shutil.move(dir, img_dir)


structure_validation_images("data/tiny-imagenet-200/val")
structure_training_images("data/tiny-imagenet-200/train")
