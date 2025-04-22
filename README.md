# Camera Identification Using SPN (Sensor Pattern Noise)
## Overview
This notebook is designed to perform image denoising and forensic analysis using advanced wavelet-based techniques. The primary goal is to extract Sensor Pattern Noise (SPN) from images and use it for camera identification and forensic analysis. The notebook includes methods for noise extraction, denoising, correlation computation, and accuracy evaluation.

## Prerequisites
Before running the notebook, ensure you have the following installed:

- Python 3.7 or higher
- Required Python libraries:
  - OpenCV
  - PyWavelets
  - tqdm
  - scikit-image
  - h5py
  - matplotlib

You can install the required libraries using the following commands:
```bash
pip install opencv-python pywavelets tqdm scikit-image h5py matplotlib
```
## Data
Some sample data is provided in the repo under /DIP DATA

The main data on which the notebook is based is a subset of Deresden dataset available at [Dresden Dataset](https://www.kaggle.com/datasets/micscodes/dresden-image-database).

The subset can be downloaded from [Here](https://drive.google.com/file/d/14z3Z7ZCO8vnzvTMnwyIPoIBGOv--H5Rw/view?usp=sharing)

The subset consist of seven Cameras having 300 photos each.

- Canon_Ixus70_0          $\hspace{0.5in}$ : 300 Photos
- Casio_EX_Z150_0        $\hspace{0.45in}$ : 300 Photos
- FujiFilm_FinePixJ50_0  $\hspace{0.25in}$ : 300 Photos
- Nikon_D70              $\hspace{0.75in}$ : 300 Photos
- Olympus_mju_1050SW     $\hspace{0.15in}$ : 300 Photos
- Panasonic_DMC_FZ50      $\hspace{0.2in}$ : 300 Photos
- Sony_DSC_W170          $\hspace{0.45in}$ : 300 Photos

## Running the Notebook
1. Open the notebook `Main NoteBook.ipynb` in Jupyter Notebook or Jupyter Lab.
2. Execute the cells sequentially to:
   - Install necessary libraries.
   - Import required modules.
   - Define paths and parameters.
   - Perform image denoising and noise extraction.
   - Compute correlations and evaluate accuracy.

## **What is Sensor Pattern Noise (SPN)**
   - SPN is a unique noise pattern inherent to each camera sensor.
   - The notebook extracts SPN from images and uses it for camera identification.
   - The notebook discusses 3 versions of the Noise extraction as metioned below

### 1. Using the builtin funtion of scikit-image library
Using the ```denoise_wavelet()``` function to generate denoised images.

```
denoised_image = denoise_wavelet(
        noisy_image,
        method='VisuShrink',
        mode='soft',
        sigma=sigma_est,
        wavelet=wavelet,
        wavelet_levels=level,
        rescale_sigma=True
    )
```
subtracting it to  obtain the noise

### 2. Using method given in paper
 -  “Low-complexity image denoising based on 
statistical modeling of wavelet coefficients,” by M. K. Mihcak, I. Kozintsev, K. Ramchandran, and P. Moulin - 1999

### 3. Using another method (BEST RESULTS)
This method is mentioned in the paper : 

This method is pretty much similar to the method mentioned in the given paper but the only edge is that this paper suggested using something called ```spectrum eualization()```

more details in the report.



## **Correlation Computation**
   - Pearson correlation is used to compare SPN from test images with target SPN.
   - The function `compute_correlations` calculates correlations for all test images.

   $$r = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_{i=1}^{n} (x_i - \bar{x})^2 \sum_{i=1}^{n} (y_i - \bar{y})^2}}$$

##  **Accuracy Evaluation and Visualization**
   - The function `compute_accuracy` evaluates the accuracy of camera identification based on SPN correlations.
   - Results are presented as accuracy percentages for each camera.
   - The notebook includes functions to visualize results, such as correlation plots and denoised images.
   - Example: `plot_images` displays noisy, denoised, and noise images side by side.

## File Structure
- `DIP Data/`: Contains input images categorized by camera brands.
- `noise_data.h5`: Stores extracted noise data in HDF5 format.
- `denoised_noise_data.h5`: Stores noise extracted from denoised images.

## Key Functions
- `save_noise`: Extracts and saves noise from images.
- `target_noise`: Generates target SPN by averaging noise from multiple images.
- `compute_correlations`: Computes correlations between test image noise and target SPN.
- `compute_accuracy`: Evaluates accuracy of camera identification.
- `adaptive_wavelet_denoise`: Implements adaptive wavelet-based denoising.
- `spectrum_equalization`: Enhances noise patterns for better analysis.

## Notes
- Ensure the input images are organized in the `DIP Data/` directory as expected by the notebook.
- Modify paths and parameters as needed to suit your dataset.
- The notebook includes multiple methods for denoising and noise extraction. Experiment with different methods to achieve the best results.

## References
- Mihcak, M. K., Kozintsev, I., Ramchandran, K., & Moulin, P. (1999). Low-complexity image denoising based on statistical modeling of wavelet coefficients. *IEEE Signal Processing Letters, 6*(12), 300–303.