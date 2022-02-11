# Project Description

DESCRIPTION
This is a tool to scrape and clean datasets for classification from google images, classify those images
using transfer learning, and then diagnose the model performance using
confusion matrices and grad cam. And it is all built into an easy to use app
so that machine learning novices can quickly gain experience.

There are two novelties in this application: 
The first is that users can collect large datasets for image recognition tasks
with minimal effort using this application. The second
is that the confusion matrices produced after training are interactive,
meaning that users can click on any corner of confusion matrices and apply
grad-cam, score-cam, and saliency maps to images in that subset. This can be especially
useful for diagnosing model failures and understanding the inner workings of a
trained neural network. 
## Running App
```
python app/app.py
```

## Environment Setup 
INSTALLATION
1) Verify you have a version of python < 3.9 on your device, like python=3.7.6 on your
Windows PC or MAC
2) Download Chrome from the internet (for the scraping tool)
3) pip install -r requirements.txt


EXECUTION
```
1) python app/app.py
```

Additional Instructions
Once inside the app you can upload the dataset.zip class as a test and train your
model. See the video for additional instructions.

DEMO VIDEO
https://youtu.be/zRKHzcFV0KQ
© 2022 GitHub, Inc.
Terms


## Disclaimers
* Using this tool for personal profit is probably risky since the images scraped may be copyrighted.
* Please be respectful and do not use this tool to discrimate, harm, or harass. Let us be scientists - not mad scientists. 


## Code Sources:
* https://github.com/debadridtt/Scraping-Google-Images-using-Python


