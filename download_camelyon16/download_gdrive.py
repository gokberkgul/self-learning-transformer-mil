'''
This module is mostly copy pasted from: https://github.com/chentinghao/download_google_drive

Guide for usage:
In your terminal, run the command:

python download_gdrive.py /path/to/Camelyon16

Credited to 
https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive
author: https://stackoverflow.com/users/1475331/user115202
'''

import sys
import os
import requests
import pandas as pd
from tqdm import tqdm

def download_file_from_google_drive(id, destination):
    def get_confirm_token(response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024) as bar:
                for chunk in response.iter_content(CHUNK_SIZE):
                    if chunk:  # filter out keep-alive new chunks
                        f.write(chunk)
                        bar.update(CHUNK_SIZE)

    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    


if __name__ == "__main__":
    if len(sys.argv) is not 2:
        raise ValueError("Usage: python google_drive.py /path/to/Camelyon16")
    path_to_camelyon = sys.argv[1]
    if not os.path.exists(path_to_camelyon):
        os.makedirs(path_to_camelyon)
    #: Keep a list of downloaded slides to continue where it left from
    downloaded_slides = open('downloaded_slides.txt', 'a+')
    downloaded_slides.seek(0)
    downloaded_slides_arr = downloaded_slides.read()
    downloaded_slides_arr = downloaded_slides_arr.split('\n')

    #: Normal slides
    path_to_normal = os.path.join(path_to_camelyon, 'training', 'normal')
    if not os.path.exists(path_to_normal):
        os.makedirs(path_to_normal)
    normal_data = pd.read_csv('normal.csv')
    for name, file_id in zip(normal_data['Name'], normal_data['File-Id']):
        if name in downloaded_slides_arr:
            continue
        print(name)
        download_file_from_google_drive(file_id, os.path.join(path_to_normal, name))
        downloaded_slides.write(name + '\n')
        downloaded_slides.flush()

    #: Tumor Slides
    path_to_tumor = os.path.join(path_to_camelyon, 'training', 'tumor')
    if not os.path.exists(path_to_tumor):
        os.makedirs(path_to_tumor)
    tumor_data = pd.read_csv('tumor.csv')
    for name, file_id in zip(tumor_data['Name'], tumor_data['File-Id']):
        if name in downloaded_slides_arr:
            continue
        print(name)
        download_file_from_google_drive(file_id, os.path.join(path_to_tumor, name))
        downloaded_slides.write(name + '\n')
        downloaded_slides.flush()
    
    #: Test Slides
    path_to_test = os.path.join(path_to_camelyon, 'testing', 'images')
    if not os.path.exists(path_to_test):
        os.makedirs(path_to_test)
    test_data = pd.read_csv('test.csv')
    for name, file_id in zip(test_data['Name'], test_data['File-Id']):
        if name in downloaded_slides_arr:
            continue
        print(name)
        download_file_from_google_drive(file_id, os.path.join(path_to_test, name))
        downloaded_slides.write(name + '\n')
        downloaded_slides.flush()
