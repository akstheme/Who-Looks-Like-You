# Who-Looks-Like-You
This MATLAB implementation allows you to find similar persons based on facial patterns. It utilizes deep learning models for feature extraction and similarity search.
# Who Looks Like You

This MATLAB implementation allows you to find similar persons based on facial patterns. It utilizes deep learning models for feature extraction and similarity search.

## Features

- Uses pre-trained deep learning models (e.g., SqueezeNet, VGG-16, VGG-19)
- Extracts features from images using the deep learning model
- Computes cosine similarity between the test image and dataset images
- Displays the top 5 most similar images

## Requirements

- MATLAB R2021a or later
- Deep Learning Toolbox
- Pre-trained CNN model (e.g., SqueezeNet)

## Usage

1. Clone the repository:
    ```sh
    git clone https://github.com/your_username/Who-Looks-Like-You.git
    ```
2. Navigate to the repository directory:
    ```sh
    cd Who-Looks-Like-You
    ```
3. Add your dataset to the specified directory.
4. Update the `face_verification.m` script with the path to your test image.
5. Run the script in MATLAB:
    ```matlab
    run('face_verification.m')
    ```

## Directory Structure

