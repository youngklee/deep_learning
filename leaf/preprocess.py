import glob
import PIL
from PIL import Image

#
# Scale an image while keeping the aspect ratio. Pad the image with the black
# background.
#
def resize_image(img, target_width, target_height):
    width, height = img.size

    if target_width - width < target_height - height:
        new_width = target_width
        new_height = int(target_width*height/width + 0.5)
    else:
        new_height = target_height
        new_width = int(target_height*width/height + 0.5)

    resized_img = img.resize(
            (new_width, new_height), resample = PIL.Image.BICUBIC)
    padded_img = Image.new('1', (target_width, target_height), 0)
    top_left_corner = (
            (target_width - new_width)//2, (target_height- new_height)//2)
    padded_img.paste(resized_img, top_left_corner)

    return padded_img

#
# Find max width and max height among the images
#
def find_max():
    max_width = 0
    max_height = 0

    for filepath in glob.glob('./images/*.jpg'):
        img = Image.open(filepath)
        width, height = img.size
        if width > max_width:
            max_width = width
        if height > max_height:
            max_height = height

    return max_width, max_height

#
# Resize images and save them in a new directory
#
def resize_images(width, height):
    for filepath in glob.glob('./images/*.jpg'):
        img = Image.open(filepath)
        filename = filepath.split('/')[-1]
        resized_img = resize_image(img, width, height)
        resized_img.save('./resized_images/' + filename, dpi = (1, 1))


width, height = find_max() # (width, height) = (1710, 1090)
width = 224
height = 224
resize_images(width, height)
