# Overview

In order to classify the ordinary Raman spectra collected from formaldehade-soaked and water-soaked shrimp surfaces, we adopted the InceptionTime deep learning model. The normal Raman spectrum data set of prawn surface was established. Shrimps were soaked in different solutions (0, 0.1, 0.25, 0.5, 1, and 2 mol/L) for 1 h, wiped clean, and the Raman spectra of their surfaces were collected directly. For each shrimp sample, spectra were collected from at least 30 random points for each surface (Fig. 1A), spanning the range of 300 - 3500 cm⁻¹. We collected a total of 482 surface Raman spectra of shrimps, resulting in a dataset with 19,504 Raman spectra. The core architecture of the model consists of three inception, each of which has four convolutional layers of different scales to extract spectral features from multiple angles. Finally, it was successfully applied to the identification of Raman spectra of prawns with illegal formaldehyde addition.

# System Requirements

## Hardware Requirements

The server hardware parameters used in the model are:

'''
CPU: Intel Xeon Gold 6240@2.6GHz
GPU: Nvidia GeForce RTX 2080 Ti
'''

## Software Requirements

This package depends on the Python scientific stack, and the versions used for testing are:

```
Python 3.7
PyTorch 1.10
Pandas 1.1.5
Numpy 1.21.2
Scipy 1.7.3
Scikit-learn 1.0.2
Matplotlib 3.5.1
Shap 0.42.1
Xgboost 1.6.2
Xlsxwriter 3.1.0
```

# File description

File "dataset(10%).csv" contains 10% normal Raman spectral data collected from formaldehade-soaked and water-soaked shrimp samples.

'''
"Shrimp_id" indicates the number of the prawns collected
"Label" indicates formaldehyde residue category (0 is negative, 1 is positive)
"FA_value" is the specific residual amount of formaldehyde measured by spectrophotometry
"Group" indicates the classification group of formaldehyde residue: '≤ 5': 0, '5~100': 1, '100~500': 2, '500~1000': 3, '≥ 1000': 4
"Raman_spectrum" Indicates the number of the collected Raman spectrum
Starting with column 5 of the data table (numbered from 0) are Raman spectral data, with a total of 2048 data points
'''

## How to use the package

1. Place "inception_raman.py" and ""Cmp_CNN.py"" in the "shrimp_class" sibling directory to ensure normal calls.

2. Run the file "shrimp_class.py" to get classification accuracy and other evaluation indexes of different models for data sets. The data of different formaldehyde immersion concentration groups can be selected according to the task requirements.

3. Run the file "shap.py" to get the shap value for the inceptiontime model.

4. File "pretreatment.py" contains the spectral preprocessing methods used in this paper.
