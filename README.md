# BugBot
Image classification of common household pest images

# About BugBot:
Household pest identification remains a significant challenge for New England residents, impacting individuals and communities such as property owners, renters, and residential building managers. Our proposed convolutional neural network (CNN) model for BugBot addresses this need through automated classification of common household pests. Unlike existing agricultural-focused solutions, our proposed model processes images captured under varied conditions, including diverse lighting and backgrounds. This diverse image approach offers a more robust classification than current models which heavily rely on controlled laboratory images and are irrelevant to everyday residents due to the agricultural focus of the research. 

# BugBot Context
Recent studies have demonstrated the effectiveness of CNNs in pest classification across various contexts and further reveal a consistent emphasis on data processing to improve model performance. Our project aims to bridge the gap between these agricultural-focused studies and household pest identification needs by incorporating the successful methodological elements identified in the literature including image rotation and mirroring. Our planned use of web-scraped images and potential ensemble method investigation represents a novel approach that addresses everyday people’s needs while building upon established technical foundations.

# Purpose
The purpose of BugBot is to assist New England residents in identifying common household pests using deep learning techniques. By utilizing a balanced and diverse dataset of insect images, BugBot provides a reliable, accurate, and user-friendly tool that helps users classify insects in their homes quickly and confidently. This eliminates the hassle of misclassifying insects or spending time searching for information on the internet, offering a streamlined and efficient solution to pest identification. 

# Goals
1. Diverse Data: Ensure a balanced and diverse dataset through rigorous data collection and preprocessing. 
2. User-Friendly Deployment: Provide an intuitive interface through Streamlit for easy access and use.
3. Accurate Classification: Classify 11 common household pests using a deep learning model and obtain high accuracy.

# Our Dataset
The data collection process is key to achieving our goal and provides a strong and consistent foundation for our project because of its diverse and evenly represented classes. 

The majority of the BugBot dataset was scraped using a publicly available API for scraping images from Bing search queries. A manual review and filtering process was then applied to the Bing-based dataset and was subject to the following standards:
- The insect must be present in the image 
- The image must show at least 75\% of the insect’s body 
 The photo is not a drawing, cartoon, or an AI generated image
- The image depicts the adult/matured insect

For all collected images, it is also essential to eliminate duplicates or near-duplicates to prevent data leakage. Visual inspection techniques will be used to verify this in addition to image hashing as needed. Since no temporal or spatial dependencies exist in the dataset, random splitting for training, validation, and test subsets is appropriate. The validation set will guide hyper-parameter tuning, while the test set will provide an unbiased assessment of the model’s performance.

Furthermore, it is important to note the unique nature of many pests that usually infest home environments in groups or colonies, including ants, termites, and bed bugs. Therefore, a direct effort was made to collect additional group images of insect infestations to be included in the dataset. After filtering, a range of approximately 16\%-76\%, depending on the insect class, was kept from the original scraped data based on the above criteria. 

After web scraping, additional data was collected by performing manual Google searches for image results using insect keywords, and by taking advantage of community postings on websites such as Reddit to collect home-environment-specific images. After combining the scraped images and manual supplement, the resulting dataset includes 160 raw images across each of the 11 insect classes.

Since our dataset consists of RGB images with x and y coordinates, the only features are the image pixels themselves. However, additional hidden features will be extracted through preprocessing and modeling the data and includes visual characteristics, such as texture, RBG value, and pixel sequence. Furthermore, when the complete image matrix is flattened into a 1-dimensional feature vector, each pixel element becomes a feature of the overall image. 

The target variable in this supervised learning project is the insect class, representing one of the 11 common household pests. It is a well-defined categorical variable with clear labels derived from the dataset. Since the dataset is designed with a balanced distribution of 160 images per class (100 for training, 40 for validation, and 20 for testing), there is no inherent class imbalance, and resampling techniques are not necessary. This ensures that the model will not favor any single class during training or evaluation.

Due to the dataset design and manual filtering criteria, we have curated a dataset that adequately represents every insect class to ensure equal distribution of training, validation, and test data to reduce bias. We are also aware of potential bias in web scraping data due to the prevalence of scientific images which may not reflect the targeted end user – an everyday homeowner classifying an insect in their home. To mitigate this we have further supplemented the dataset with manually selected images with diverse backgrounds and contexts including images from Reddit and YouTube thumbnails.

In terms of resources required, the dataset, consisting of 1,760 images is relatively small and is manageable to process with standard resources. The dataset size, even after augmentation, is unlikely to exceed the limits of local storage or memory. Therefore, the dataset does not require advanced techniques such as distributed processing (e.g., using Apache Spark or Dask) since the computational and storage demands are minimal. 
Although no distributed processing will be required, the data will require multiple preprocessing steps which include:

- Standardizing the data to a fixed image size (e.g., 224x224)
- Removing duplicate data via image hashing
- Normalizing the pixel values
- Adding padding to the images
- Applying data augmentation techniques including rotation, height and width shift, brightness adjustments, and zoom, to expand the training dataset

It is important to acknowledge the limitations of this dataset which could include a lack of sufficient data, potentially leading to over-fitting during training and data imbalance. When this case occurs, it could skew model predictions and affect the model's accuracy. This will be addressed and resolved through data augmentation, mentioned above, through techniques including image rotation, horizontal and vertical flips, and cropping of images. By ensuring that each insect class contains 160 unique images and by expanding the dataset to include varied versions of all images, we effectively diversify the data set to avoid over-fitting and enhance the model performance.

The final outcome of BugBot will provide interpretable insights to end users by providing accurate classifications of pests in their homes. We will utilize Streamlit to deliver a seamless user-friendly interface that takes in a user’s uploaded insect image and communicates the model output  – the predicted classification – through a text display on the screen. By using Streamlit as a platform for our model deployment, users will be able to receive their results in real time.

# Setup Instructions

Pip install requirements.txt to a virtual environment (Python 3.12)

1. **cd** to repo location (change directory)

Complete the following if you do not already have an environment set up with our requirements.txt:

2. Type: **pip install virtualenv**, press enter.
3. Update pip if needed (type: **pip install --upgrade pip**, press enter).
4. Type: **python -m venv bugbot_env**, press enter
5. Type: **source bugbot_env/bin/activate**, press enter
6. You should now see something that looks like **(bugbot_env)(base) ** in front of your curse in the terminal
7. Type: **pip install -r requirements.txt**, press enter
8. To run the notebook you are interested in, type jupyter notebook in the terminal of your now activated environment. After the browser opens, click the notebook of interest.
9. To run the preprocessing script, download the raw data in DATA provided in the repo. Type: **python data_processing_pipeline.py**
10. To run the tuner scripts, first run the data processing pipeline in step 9. Then navigate to your terminal, and ensure you are in the bugbot-main directory. To run a tuner script, type the following:

    **python "Model Tuning Scripts/{model}_run_tuner_script.py" --epochs {some int} --patience {some int} --min_delta {some float} --executions_per_trial {some int} --max_trials {some int}**

    --> replace content with '{}' with the values you are hoping to run
    --> example: **python "Model Tuning Scripts/DenseNet201_run_tuner_script.py" --epochs 20 --patience 3 --min_delta 0.001 --executions_per_trial 1 --max_trials 20**
    
12. Done!




