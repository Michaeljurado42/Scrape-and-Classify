# Project Description
This lightweight tool makes the machine learning image classification easy. It allows you to specify classes and download images from those classes from the internet. Then you can run a lighweight training script and train a classifier to differentiate images from those classes.

## Running App
```
python app/app.py
```

## Environment Setup 
This project works with python3 environments less than python3.9
### manual setup
    dash==2.0.0
    dash_bootstrap_components==1.0.0
    dash_core_components==2.0.0
    dash_html_components==2.0.0
    keras==2.7.0
    matplotlib==3.1.3
    numpy==1.18.1
    pandas==1.0.1
    Pillow==8.4.0
    plotly==5.4.0
    random_word==1.0.7
    requests==2.22.0
    scikit_learn==1.0.1
    scipy==1.4.1
    selenium==4.1.0
    tensorflow==2.2.0
    tf_keras_vis==0.8.0
### conda setup
    conda env create -f environment.yml
    conda activate instant_learning
The dataset application requires chromedriver.exe to be placed in the repository folder to work

## 1: Download Custom Dataset From Google Images
    python google_image_scraping_script.py <class1>, <class2>, ... <classn> --num_images <number of images for each class>
This script creates a dataset directory with class subdirectories. Run with "--help" 
for more information

## 1.1: Cifar-10 instructions
    Download dataset from https://github.com/YoongiKim/CIFAR-10-images/tree/master/train and extract to dataset folder
    Example Project structure
    Scrape-And-Classify
        dataset
            airplane
            automobile
            bird
            cat
            deer
            dog
            frog
            horse
            ship
            truck

## 2: Train Classifier
    python transfer_learning.py <class1>, <class2>, ... <classn> --model_type <string identifier>
This script creates training, validation, and testing partitions based on the data downloaded in part 1. Then it trains a machine learning classifier on those classes. The classifier is exported as trained_model.hd5. Run with "--help" for more information

## Full Example 1
To create a DNN classifier capable of differentiating dogs from cats we run the following scripts
##
    python google_image_scraping_script.py dog cat --num_images 1000
    python transfer_learning.py dog cat --model_type mobilenet

## Full Example 2
To create a DNN classifier capable of differentiating hammers from everything else in the world we can run
##
    python google_image_scraping_script.py hammer random --num_images 1000
    python transfer_learning.py hammer random --model_type mobilenet    

## Repository Features
* Uses data augmentation during training 
* Uses pretrained models from keras.applications for transfer learning
* Prints out testing accuracy at the end of training
* Uses early stopping to prevent overfitting
* saves learning_curve.png after training

![Alt text](graphics/machine_learning.png?raw=true "Title")
## Limitations
* To obtain good classification performance it may be neccessary to weed out "bad examples" after running google_image_scraping_script.py
* In order to properly download images from google you cannot minimize the chrome application that is automatically opened by google_image_scraping_script.py.
* random images are extracted currently by searching google with random nouns and verbs and extracting images. Since there are many obscure words in the english language this process produces many weird (and sometimes inappropriate) images
* It takes ~20 minutes to obtain 1000 images for a single class on my pc. For large projects, I recommend to put your pc into low power mode at night and let the dataset collection process work while you sleep.

## Things to be Added
* Greater customization of training hyperparams at the command line

## Disclaimers
* Using this tool for personal profit is probably risky since the images scraped may be copyrighted.
* Please be respectful and do not use this tool to discrimate, harm, or harass. Let us be scientists - not mad scientists. 


## Code Sources:
* https://github.com/debadridtt/Scraping-Google-Images-using-Python


