# Melanoma_detection_comp_vision
This study explores the application of deep learning techniques, specifically transformers, in computer vision tasks for image classification, focusing on the crucial domain of melanoma detection

This project has been made in order to explore the power of AI in the healthcare field, and show its potential as a tool in assisting doctors in disease detection. Melanoma, a type of skin cancer, presents a significant public health challenge worldwide. Early and accurate detection of melanoma can significantly improve patient outcomes. In this research, we investigate the effectiveness of utilising pre-trained transformer models for classifying skin lesions as benign or malignant based on image data. Through extensive experimentation and optimization, we aim to develop a robust and reliable model for melanoma detection, contributing to the advancement of computer-aided diagnosis systems in dermatology.

## Introduction
Early detection plays a pivotal role in improving patient prognosis and survival rates. Traditional methods of diagnosing melanoma rely heavily on visual inspection by dermatologists, which can be subjective and prone to error. With the rapid advancements in deep learning and computer vision, there is growing interest in developing automated systems for melanoma detection using machine learning techniques.

In recent years, deep learning models, particularly convolutional neural networks (CNNs), have demonstrated remarkable success in image classification tasks, including medical image analysis. However, the emergence of transformer-based architectures, originally designed for natural language processing tasks, has sparked a new wave of innovation in computer vision.

In this study, we leverage the power of transformers to address the challenge of melanoma detection from dermatoscopic images. By fine-tuning pre-trained transformer models on a dataset of skin lesion images labelled as benign or malignant, we aim to develop an accurate and efficient classification model. The ultimate goal is to create a reliable tool that can assist dermatologists in early diagnosis and decision-making, ultimately leading to improved patient outcomes and better management of skin cancer.

## The Dataset
Our chosen dataset can be found on Kaggle, with the following URL: https://www.kaggle.com/datasets/bhaveshmittal/melanoma-cancer-dataset

It contains images uniformly sized at 224 x 224 pixels, each offering a comprehensive view of melanoma's diverse manifestations. The data is split into a train and test set. The train contains 6289 images representing benign lesions and 5590 malignant images. On the other hand, the test set contains 1000 images of each class.

## Preprocessing the Data
This dataset was chosen due to the fact that it is balanced and clean, and the high number of images it contains compared to other data sources found. The preprocessing steps involve converting images to tensors, and loading them into training and validation datasets. These transformations ensure uniformity and compatibility with PyTorch models. Additionally, data loaders are created to efficiently handle batch processing and parallel loading, enhancing training and validation performance.

## Type of Models Chosen
We chose to focus on deep learning approaches over classical machine learning models. We did so because deep neural networks will allow us to achieve better results, even though it requires more computing power. But since training a deep neural network from scratch requires tremendous amounts of data, we opt for transformers. They yielded fairly good results, but the downside being runtime and computing capacity.

## Metric of Interest
When choosing the metric for evaluation, we initially thought of accuracy, thus maximising the amount of images predicted correctly overall. But on second thought, it is preferable to optimise recall in this case, because the consequences of wrongly classifying a malignant image as benign can mean life or death. Thus, it is absolutely vital to minimise false negatives, and this is what we focused on in our code.

## Fine Tuning
For fine tuning the model we used Optuna. To fine tune the data we had to split the train set into train and validation sets. We defined an ‘objective’ function representing the recall metric that Optuna tries to optimise. Within each trial, the function trains the model for a certain number of epochs using the hyperparameters sampled for that trial. However, this definitely added a lot of runtime and necessary GPU. One must take this into account when attempting to fine-tune a large pre-trained model.

We chose to run the same batch size and epochs to get a feel of which pre-trained model worked best. Once we found the best pre-trained model, we tried different batch sizes, epochs and trials to see what yielded the best scores.

| Model_name          | Epochs, Batch_size | Learning Rate | Weight Decay | Trials | Recall  |
|---------------------|--------------------|---------------|--------------|--------|---------|
| Resnet18            | 10, 32             | 0.00046       | 0.000146     | 5      | 0.899   |
| Resnet50            | 20, 16             | 0.000083      | 0.000174     | 5      | 0.9620  |
| vit-base-patched    | 10, 32             | 0.000804      | 0.000002     | 5      | 0.9110  |
| deit-base-distilled | 10, 32             | 0.000117      | 0.000039     | 5      | 0.9597  |



## Conclusion
As you can see, the fine-tuned Resnet50 yielded the best results. We thus chose it as our final model. Out of 2000 images in the test set, 1926 were classified accurately and only 38 out of 1000 malignant images were miss-classified.

## Recommendations for Future Work
Regarding future work, I would advise to increase GPU capacity, in order to handle batch sizes that require more capacity, and also to train more epochs and trials. The difficulty when using transformers comes down to computing power. I would also advise to try different methods to optimise runtime, which is something we did not necessarily spend a lot of time on. Additionally, it would be interesting to explore other architectures, such as Vision Transformers, which have shown promising results in image classification tasks. Finally, it would be beneficial to test the model on a larger and more diverse dataset, in order to further validate its performance.
