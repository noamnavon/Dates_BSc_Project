import os


def collect_image_paths(root_folder, extensions=None):
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".jp2"}
    return [
        os.path.join(dirpath, filename)
        for dirpath, _, filenames in os.walk(root_folder)
        for filename in filenames
        if os.path.splitext(filename)[1].lower() in extensions
    ]