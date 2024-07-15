import os
import requests
from PIL import Image
from io import BytesIO

# Constants
BASE_URL = "https://api.isic-archive.com/api/v2"
SEARCH_URL = f"{BASE_URL}/images/search/"
LIMIT = 500  # image per category
PAGE_SIZE = 100  

BENIGN_PATH = r"C:\Users\Joseph Moubarak\Desktop\OMANNA SAD\dataset\benign"
MALIGNANT_PATH = r"C:\Users\Joseph Moubarak\Desktop\OMANNA SAD\dataset\malignant"

#mk dir for dataset
os.makedirs(BENIGN_PATH, exist_ok=True)
os.makedirs(MALIGNANT_PATH, exist_ok=True)

def download_image(image_url, image_id, label):
    response = requests.get(image_url)
    
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content))
        if label == 'benign':
            img.save(os.path.join(BENIGN_PATH, f"{image_id}.jpg"))
        elif label == 'malignant':
            img.save(os.path.join(MALIGNANT_PATH, f"{image_id}.jpg"))
        print(f"Downloaded {image_id} as {label}")
    else:
        print(f"Failed to download {image_id}")

def search_and_download_images(query, label, limit=LIMIT, page_size=PAGE_SIZE):
    downloaded_count = 0
    next_url = SEARCH_URL
    
    while downloaded_count < limit and next_url:
        params = {'query': query, 'limit': page_size}
        response = requests.get(next_url, params=params)
        
        if response.status_code == 200:
            images = response.json()
            for image in images['results']:
                if downloaded_count >= limit:
                    break
                image_id = image['isic_id']
                image_url = image['files']['full']['url']
                download_image(image_url, image_id, label)
                downloaded_count += 1
            next_url = images.get('next')
        else:
            print(f"Failed to retrieve images for query: {query}")
            break

def main():
    benign_query = "benign_malignant:benign"
    malignant_query = "benign_malignant:malignant"
    
    print("Starting download of benign images...")
    search_and_download_images(benign_query, 'benign')
    
    print("Starting download of malignant images...")
    search_and_download_images(malignant_query, 'malignant')

if __name__ == "__main__":
    main()
