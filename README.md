# Face Mask Detection

## Overview
This project implements a face mask detection algorithm that identifies whether a person is wearing a mask in a given image. The algorithm utilizes a pre-trained CNN model (VGG16) to classify images into two categories: "With Mask" and "Without Mask." This project aims to assist in monitoring mask compliance in public spaces while maintaining high accuracy and generalization.

---

## Dataset
The dataset used for this project can be found on [Kaggle](https://www.kaggle.com/datasets/omkargurav/face-mask-dataset.)
- **Dataset location**: Place the downloaded ZIP file in the `FaceMaskDataSet` directory.
- **Extraction**: Use the provided Python script to extract the dataset contents.
```python
import zipfile

zip_path = os.path.join('FaceMaskDataSet', 'archive.zip')
dataset_path = os.path.join('FaceMaskDataSet')

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(dataset_path)
```

 includes images of faces categorized into two folders:  
- **`with_mask`**: Images of individuals wearing masks.  
- **`without_mask`**: Images of individuals not wearing masks.

The dataset should be placed under the `FaceMaskDataSet` directory.

## Requirements
**Python Version**
  - This has been **trained and tested** on Python  **3.11.3**.
  - Ensure you have Python 3.11 or higher installed for optimal compatibility.

To install the required dependencies, run:
```bash
pip install -r requirements.txt
```

**Dependencies:**
- `numpy==1.26.4`
- `pandas==2.1.2`
- `matplotlib==3.9.1`
- `sklearn==1.3.2`
- `tensorflow==2.13.1`
- `Pillow==10.4.0`

---

## Cloning the Repository
To clone this repository onto your local machine:
1. Use the following command:
```bash
git clone https://github.com/FarisAnsara/FaceMaskDetection.git
```
2. Navigate to the project directory:
```bash
cd FaceMaskDetection
```

---

## How to Run
### 1. **Setting Up the Environment**
   - Ensure you have Python 3.11.3 installed.
   - Install the dependencies using:
     ```bash
     pip install -r requirements.txt
     ```

### 2. **Running the Detector**
   - To run the face mask detection script, use the following command:
     ```bash
     python main.py --image_path <path_to_input_image>
     ```
   - Replace `<path_to_input_image>` with the path to the image you want to test.

   - If no arguments are provided, the script uses the default image and model paths:
     ```bash
     python main.py
     ```
     - Default input image: `FaceMaskDataSet/test_image/with_mask_5.jpg`

---

## Documentation
If you want detailed analysis, see the `FaceMaskDetection.ipynb` Jupyter notebook.
### **Model Architecture**
- **Pre-trained Model Used**: VGG16  
- **Training Dataset**: Images categorized into `with_mask` and `without_mask`.  
- **Approach**: The VGG16 base model was used with additional fully connected layers to adapt the model for binary classification.

### **Testing**
- The model was evaluated on a separate test dataset, achieving **96% accuracy**, with detailed metrics provided (e.g., precision, recall, and F1-score).

---

## Results
- **Training Results**: The training process achieved convergence with no signs of overfitting.
- **Testing Results**: Key metrics such as confusion matrix and precision-recall curves were plotted to validate performance.  
- **Performance Highlights**:
  - Accuracy: **96%**
  - Precision: **96%**
  - Recall: **96%**


---

## Future Enhancements
1. Expand the dataset to include more real-world variations, such as occlusions and different lighting conditions.
2. Optimize the model for faster inference on mobile and embedded systems using TensorFlow Lite.
3. Explore alternative architectures like EfficientNet or YOLOv8 for enhanced performance.
4. Implement support for live video stream analysis in real time.

---

## Notes
- This project is designed to work seamlessly on both Windows and Linux platforms.
- Ensure the dataset and model paths are correctly specified in the script or passed as arguments.
- For any issues, feel free to open an issue on the [GitHub repository](https://github.com/FarisAnsara/FaceMaskDetection).
