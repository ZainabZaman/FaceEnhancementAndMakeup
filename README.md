# Face Restoration with CodeFormer

This repository contains a Python script for face restoration using the CodeFormer model. The script leverages face recognition and CodeFormer, a deep learning model, to enhance facial features in images.

## Prerequisites

Before running the script, make sure you have the following dependencies installed:

- Python
- OpenCV (`pip install opencv-python`)
- PyTorch (`pip install torch torchvision`)
- NumPy (`pip install numpy`)
- basicsr library (Ensure that the required libraries are installed. You can find them in the `requirements.txt` file.)

## Usage

1. Clone the repository:

    ```bash
    !git clone https://github.com/sczhou/CodeFormer.git
    ```

2. Install the required Python packages:

    ```bash
    pip install -r requirements.txt
    ```
    or through the provided `CodeFormer.ipynb` nootebook.
   
3. Replace the `./CodeFormer/inference_codeformer.py` with the `inferece_codeformer.py` file provided in this repository.
  
4. Run the `inference_codeformer.py` script script:

    ```python
    python face_restoration.py
    ```
## Options

The script includes options for different makeup styles:

- `option_makeup = 1`: Deep Gray Eyebrows, Red Lip, Gray Eyes, Black Eyeliner
- `option_makeup = 2`: Brown Eyebrows, Hot Pink Lip, Gray Eyes, Brown Eyeliner
- `option_makeup = 3`: Deep Gray Eyebrows, Dark Orange Brown Lip, Gray Eyes, Black Eyeliner
- `option_makeup = 4`: Deep Gray Eyebrows, Light Pink Lip, Gray Eyes, Brown Eyeliner
- `option_makeup = 5`: Deep Gray Eyebrows, Crimson Lip, Gray Eyes, Black Eyeliner

You can modify the `option_makeup` variable in the `inferece_codeformer.py` script to choose a specific makeup style.

## Examples
The following were the results from the DLIB makeup application code
![dlib_results](https://github.com/ZainabZaman/FaceEnhancementAndMakeup/blob/08a563cfbea6b3b612cbfd9092e218a4b7a30e9d/examples/results_makeup_dlib.png)

Below are the results from the DLIB library refined through Codeformer
![dlib_results](https://github.com/ZainabZaman/FaceEnhancementAndMakeup/blob/0030dd746c4401b59bd4e8d3453aee77305f2c0e/examples/results_makeup_refined.png)

Shown below are the final results with makeup application through DLIB and refinement through Codeformer
![dlib_results](https://github.com/ZainabZaman/FaceEnhancementAndMakeup/blob/0030dd746c4401b59bd4e8d3453aee77305f2c0e/examples/final_results.png)

## Acknowledgments

- The CodeFormer model is used for face restoration.
- Face recognition is performed using the `face_recognition` library.


