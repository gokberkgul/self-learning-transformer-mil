import hashlib
import os
from tqdm import tqdm
import sys

def discover_slides(folder):
    """Walks through all files in the given folder and subfolders, returns a list of slide paths.

    Parameters
    ----------
    folder : string
        Main folder of dataset

    Returns
    -------
    slides : list
        List of slide paths found

    """
    slides = []
    for root, _, files in os.walk(folder, topdown=False):
        for file_name in files:
            if ".tif" in file_name:
                slides.append(os.path.join(root, file_name))
    return slides

def parse_md5_checksum():
    checksums_file = open("checksums.md5", "rb")
    checksums = checksums_file.read().decode('utf8').split('\n')
    checksums = [checksum for checksum in checksums if '.tif' in checksum]  #: We will only check .tif files
    checksums = [line.split(' ')[0] + ',' + line.split(' ')[1][1:] for line in checksums]
    checksums_file.close()
    return checksums

if __name__ == '__main__':
    if len(sys.argv) is not 2:
        raise ValueError("Usage: python checksum_test.py /path/to/Camelyon16")
    path_to_camelyon = sys.argv[1]
    slides = discover_slides(path_to_camelyon)
    checksums = parse_md5_checksum()
    for slide in tqdm(slides):
        md5_hash = hashlib.md5()
        a_file = open(slide, "rb")
        content = a_file.read()
        a_file.close()
        md5_hash.update(content)
        digest = md5_hash.hexdigest()

        slide_name = slide.split('/')[-3] + '/' + slide.split('/')[-2] + '/' + slide.split('/')[-1]
        for checksum in checksums:
            if slide_name in checksum:
                md5_hash_part = checksum.split(',')[0]
                if digest == md5_hash_part:
                    break
                else:
                    raise ValueError("Checksum failed at", slide_name)
    print("Checked all .tif images, no problem found")