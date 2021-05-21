import selenium
from selenium import webdriver
from random_word import RandomWords
import time
import csv
import requests
import os
from PIL import Image
import io
import hashlib
import os
# All in same directory
DRIVER_PATH = 'chromedriver.exe'


def fetch_image_urls(query:str, max_links_to_fetch:int, wd:webdriver, sleep_between_interactions:int=1):
    def scroll_to_end(wd):
        wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(sleep_between_interactions)        
    
    # build the google query
    search_url = "https://www.google.com/search?safe=off&site=&tbm=isch&source=hp&q={q}&oq={q}&gs_l=img"

    # load the page
    wd.get(search_url.format(q=query))

    image_urls = set()
    image_count = 0
    results_start = 0
    error_clicks = 0
    while (image_count < max_links_to_fetch) & (error_clicks < 30): # error clicks to stop when there are no more results to show by Google Images. You can tune the number
        scroll_to_end(wd)

        print('Starting search for Images')

        # get all image thumbnail results
        thumbnail_results = wd.find_elements_by_css_selector("img.Q4LuWd")
        number_results = len(thumbnail_results)
        print(f"Found: {number_results} search results. Extracting links from {results_start}:{number_results}")
        for img in thumbnail_results:
            # try to click every thumbnail such that we can get the real image behind it
            print("Total Errors till now:", error_clicks)
            try:
                print('Trying to Click the Image')
                img.click()
                time.sleep(sleep_between_interactions)
                print('Image Click Successful!')
            except Exception:
                error_clicks = error_clicks + 1
                print('ERROR: Unable to Click the Image')
                if(results_start < number_results):
                	continue
                else:
                	break
                	
            results_start = results_start + 1

            # extract image urls    
            print('Extracting of Image URLs')
            actual_images = wd.find_elements_by_css_selector('img.n3VNCb')
            for actual_image in actual_images:
                if actual_image.get_attribute('src') and 'http' in actual_image.get_attribute('src'):
                    image_urls.add(actual_image.get_attribute('src'))

            image_count = len(image_urls)

            print('Current Total Image Count:', image_count)

            if len(image_urls) >= max_links_to_fetch:
                print(f"Found: {len(image_urls)} image links, done!")
                break
            else:
                load_more_button = wd.find_element_by_css_selector(".mye4qd")
                if load_more_button:
                    wd.execute_script("document.querySelector('.mye4qd').click();")
            	        
        results_start = len(thumbnail_results)

    return image_urls

def persist_image(folder_path:str,file_name:str,url:str):
    try:
        image_content = requests.get(url).content

    except Exception as e:
        print(f"ERROR - Could not download {url} - {e}")

    try:
        image_file = io.BytesIO(image_content)
        image = Image.open(image_file).convert('RGB')
        folder_path = os.path.join(folder_path,file_name)
        if os.path.exists(folder_path):
            file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        else:
            os.mkdir(folder_path)
            file_path = os.path.join(folder_path,hashlib.sha1(image_content).hexdigest()[:10] + '.jpg')
        with open(file_path, 'wb') as f:
            image.save(f, "JPEG", quality=85)
        print(f"SUCCESS - saved {url} - as {file_path}")
    except Exception as e:
        print(f"ERROR - Could not save {url} - {e}")

import os
def scrape_class(query, num_images = 200, random = False):
    if not os.path.isdir("dataset"):
        os.mkdir("dataset")

    wd = webdriver.Chrome(executable_path=DRIVER_PATH)
    wd.get('https://google.com')
    search_box = wd.find_element_by_css_selector('input.gLFyf')
    search_box.send_keys(query)
    links = fetch_image_urls(query,num_images,wd) # 200 denotes no. of images you want to download
    for i in links:
        if random:
            persist_image("dataset","random",i)
        else:
            persist_image("dataset",query,i)
    wd.quit()

import argparse

def generate_random_words(num_words: int):
    """
    generates num_words random words

    TODO: replace this with code that generates only common words
    params
    num_words: number of words to generate

    returns a list of random words

    """
    r = RandomWords()
    
    all_words = []
    batches = num_words//100
    for _ in range(int(batches)):
        all_words += r.get_random_words(includePartOfSpeech = "noun,verb", limit=100)
    remaining_words = num_words - len(all_words)
    if remaining_words != 0:
        all_words += r.get_random_words(includePartOfSpeech = "noun,verb", limit = remaining_words)
    return all_words

if __name__ == '__main__':
    p = argparse.ArgumentParser(description="scrapes google images using google chrome and puts the images into a dataset folder")
    p.add_argument('classes', nargs='*', type = str, help = "classes to scrape. (ie dog cat) If you specify 'random' as one of the classes will be a bunch of random images taken from the internet")
    p.add_argument("--num_images", type = int, help = "Number of images to scrape", default=100)
    args = p.parse_args()
    classes = args.classes
    num_images = args.num_images
    for class_ in classes:
        if class_ == "random":
            random_words = [w for w in generate_random_words(num_images//3) if w not in classes]  # ensure no duplicates

            # write random words to list
            if not os.path.isdir("dataset/random"):
                os.mkdir("dataset/random")
            with open('dataset/random/random_words.csv','w') as f:
                writer = csv.writer(f)
                writer.writerows(random_words)
            
            # scrape 3 words from each class to save time
            for word in random_words:
                scrape_class(word, 3, random=True)  # 3 per each random class
        else:
            scrape_class(class_, num_images, random=False)
