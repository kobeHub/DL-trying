import os
import requests
import zipfile


def unzip(save_path, file_path):
    with zipfile.ZipFile(file_path, 'r') as zf:
        zf.extractall(save_path)

def download_extract():
    if not os.path.exists('./ml-1m.zip'):
        print("Downloading...")
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        r = requests.get(url)
        with open('ml-1m.zip', 'wb') as f:
            f.write(r.content)
    print('Already done')
    unzip('./', './ml-1m.zip')

download_extract()

