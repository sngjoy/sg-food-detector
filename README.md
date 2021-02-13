# Singapore Food Detector
### Introduction
This project aims to classify 12 different Singapore food - chilli crab, curry puff, dim sum, ice kacang, kaya toast, nasi ayam, popiah, roti prata, sambal stingray, satay, tau huay and wanton noodle - using deep convolutional neural network. Subsequently, the model is deployed as a web application for consumption.

TensorFlow was used to build and train the CNN model. Flasks was used to build the web application.

### Basic Usage
#### Step 1:
Clone the repository 
```
git clone https://github.com/sngjoy/sg_food_detector.git
cd sg_food_detector
```
#### Step 2:
Create environment with the required packages.

```
conda env create -f conda.yml
```

#### Step 3: 
Run the app!
```
python src/app.py
```
Navigate to URL 
```
http://localhost:8000
```
## About the Model
##### Model Architecture
The model is adapted from the MobileNetV2 model. The fully connected layers from the MobileNetV2 are removed and replaced with a global average pooling layer, a dropout layer, four dense layers with ReLU activation and a final prediction layer with softmax activation.

Optimiser used was Adam with a learning rate of 0.0001. Early stopping is used based on the best validation accuracy with a patience of 15.
    
The model can be found in the `model` folder.

##### Model Parameters
Loss function: Categorical Cross Entropy
- Optimizer: Adam
- Learning Rate: 0.0001
- Batch Size: 32
- Train-Val-Test Split:
    - Training Images: 734
    - Validation Images: 490
    - Testing Images: 490

##### Model Performance
| Dataset    | Accuracy| 
| :----------|:-------:| 
| Train      | 99.2%   | 
| Validation | 96.3%   | 
| Test       | 92.6%   |

## Dataset
The base model, MobileNetV2, has been pre-trained on the ImageNet dataset.
The entire model is then trained with images from the 12 classes as mentioned. The images are split into three sets (train/validation/test) in a 60/20/20 stratified fashion. The image dimensions are 224 x 224 x 3.

Here is a breakdown of the number of images used in each class.

| Food            | No. of Images | 
| :---------------|:-------------:| 
| Dim Sum         | 171           | 
| Curry Puff      | 105           | 
| Sambal Stingray | 103           | 
| Satay           | 102           | 
| Chilli Crab     | 102           | 
| Roti Prata      | 101           | 
| Popiah          | 101           | 
| Wanton Noodle   | 100           | 
| Kaya Toast      | 100           | 
| Ice Kacang      | 90            | 
| Nasi Ayam       | 85            | 
| Tau Huay        | 64            | 

Data augmentation is used to increase the size of the dataset. In this model, shearing up to 15 degree is used to allow the model to recognise images that are taken from different angles. 

## Future Developments
- Improve on the API design

## Author
Joy Sng

## Acknowledgement
HTML Template: https://github.com/krishnaik06/Deployment-flask