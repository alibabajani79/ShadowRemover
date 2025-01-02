# Shadow Removal using U-Net

## Introduction
This project implements a shadow removal pipeline using a U-Net-based convolutional neural network. The goal is to detect shadows in input images and remove them using a predicted mask and inpainting techniques. The dataset used is the SBU Shadow Dataset.

---

## Table of Contents
- [Features](#features)
- [Dataset](#dataset)
- [Requirements](#requirements)
- [Usage](#usage)
  - [Data Preparation](#data-preparation)
  - [Training the Model](#training-the-model)
  - [Shadow Removal](#shadow-removal)
- [Results](#results)
- [References](#references)
- [License](#license)

---

## Features
- U-Net architecture for shadow detection.
- Preprocessing and resizing of images and masks.
- Shadow removal using predicted masks and OpenCV’s inpainting.
- Visual comparison of original images, predicted masks, and shadow-free outputs.

---

## Dataset
The dataset used for training and testing is the [SBU Shadow Dataset](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html). It includes:
- Shadow Images: Images containing shadows.
- Shadow Masks: Binary masks corresponding to shadow regions in the images.

The dataset is divided into training and test sets:
- Training Set: `SBUTrain4KRecoveredSmall`
- Test Set: `SBU-Test`

---

## Requirements

The project is implemented in Python. Install the required libraries using:

```bash
pip install tensorflow numpy opencv-python matplotlib
```

Ensure that you have TensorFlow 2.x and OpenCV installed.

---

## Usage

### Data Preparation
1. Download the SBU Shadow Dataset from the [official website](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html).
2. Place the dataset in your Google Drive or local directory.
3. Adjust the `zip_file_path` and `extract_to` variables to point to your dataset location.

```python
shutil.unpack_archive(zip_file_path, extract_to)
```

### Training the Model
1. Define the U-Net architecture using the `unet_model` function.
2. Compile the model with the Adam optimizer and binary crossentropy loss:

```python
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

3. Load the training images and masks using the `load_data` function.

4. Train the model with your prepared data.

### Shadow Removal
To remove shadows from an input image:
1. Use the `remove_shadow` function.
2. Input an image path and obtain the original image, predicted mask, and shadow-free image.
3. Visualize results using Matplotlib:

```python
plt.imshow(cv2.cvtColor(inpainted_image, cv2.COLOR_BGR2RGB))
```

---

## Results
After training the U-Net model, shadow detection and removal are performed successfully. Below is an example output:

1. **Original Image:** Input image with shadows.
2. **Predicted Mask:** Binary mask highlighting shadow regions.
3. **Shadow-Free Image:** Final image with shadows removed using inpainting.

---

## References
1. SBU Shadow Dataset: [https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html](https://www3.cs.stonybrook.edu/~cvl/projects/shadow_noisy_label/index.html)
2. U-Net Architecture: Ronneberger et al., 2015 ([Paper](https://arxiv.org/abs/1505.04597))
3. OpenCV Documentation: [https://opencv.org/](https://opencv.org/)

---

## License
This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

---

Feel free to contribute or raise issues for further improvements!

