import requests
import zipfile
import io
import os

def download_and_extract(url, target_dir):
    print(f"Downloading from {url}...")
    r = requests.get(url)
    if r.status_code == 200:
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(target_dir)
        print("Extracted.")
    else:
        print(f"Failed to download. Status code: {r.status_code}")

url = "https://github.com/chandrikadeb7/Face-Mask-Detection/archive/refs/heads/master.zip"
download_and_extract(url, ".")
