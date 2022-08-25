from PIL import Image


def adjust_image(image, new_width, new_height):
    # Get dimensions
    width, height = image.size
    new_size = min(width, height)

    left = int((width - new_size) / 2)
    top = int((height - new_size) / 2)
    right = int((width + new_size) / 2)
    bottom = int((height + new_size) / 2)

    # Crop the center of the image
    image = image.crop((left, top, right, bottom))
    image = image.resize((new_width, new_height), Image.LANCZOS)
    image = image.convert('RGB')

    return image
