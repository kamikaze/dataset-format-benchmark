from dataset_format_benchmark.storages import ImageFileStorage


class JPEGImageStorage(ImageFileStorage):
    IMAGE_FILE_EXTENSION = 'jpeg'
    DATASET_SUBDIR_NAME = f'{IMAGE_FILE_EXTENSION}_files'
    METADATA_FILE_NAME = 'metadata.json'

    def __init__(self, quality: int = 100):
        super().__init__()
        self.quality = quality

    def __str__(self):
        return f'{self.DATASET_SUBDIR_NAME}_{self.quality}'


class PNGImageStorage(ImageFileStorage):
    IMAGE_FILE_EXTENSION = 'png'
    DATASET_SUBDIR_NAME = f'{IMAGE_FILE_EXTENSION}_files'
    METADATA_FILE_NAME = 'metadata.json'


class BMPImageStorage(ImageFileStorage):
    IMAGE_FILE_EXTENSION = 'bmp'
    DATASET_SUBDIR_NAME = f'{IMAGE_FILE_EXTENSION}_files'
    METADATA_FILE_NAME = 'metadata.json'


class TIFFImageStorage(ImageFileStorage):
    IMAGE_FILE_EXTENSION = 'tiff'
    DATASET_SUBDIR_NAME = f'{IMAGE_FILE_EXTENSION}_files'
    METADATA_FILE_NAME = 'metadata.json'


class WebPImageStorage(ImageFileStorage):
    IMAGE_FILE_EXTENSION = 'webp'
    DATASET_SUBDIR_NAME = f'{IMAGE_FILE_EXTENSION}_files'
    METADATA_FILE_NAME = 'metadata.json'
