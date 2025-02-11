Signature Verification Using Deep Learning

Overview

This project implements a signature verification system using deep learning models. The goal is to classify whether a given signature is genuine or forged based on a dataset of handwritten signatures.

📂 Dataset

The dataset used for this project was downloaded from Kaggle. It consists of genuine and forged signatures from 30 individuals, with 5 samples per category. The images follow a naming convention: NFI-XXXYYZZZ, where:

XXX is the ID of the person who made the signature.

YY is the sample number.

ZZZ is the ID of the person whose signature appears in the image.

The dataset is split into two folders:

📁 genuine/

📁 forged/

🏗️ Models Implemented

Three deep learning models were implemented and evaluated for signature verification:

1️⃣ Convolutional Neural Network (CNN)

A basic CNN model was trained to learn feature representations of genuine and forged signatures. It consists of convolutional layers followed by fully connected layers.

2️⃣ CNN with Regularization

To improve generalization, the CNN model was trained with L2 regularization and dropout layers.

3️⃣ VGG-19 ✅

A pre-trained VGG-19 model was fine-tuned for signature verification. This model outperformed the other two, achieving the highest accuracy on the testing dataset.

🔍 Forgery Detection

A separate script was created to check for forged signatures. It loads the trained model (model.h5) and evaluates the testing dataset, predicting whether each signature is genuine or forged.

📊 Results

VGG-19 achieved the best performance among all models.

Regularization improved the CNN model's performance but was still outperformed by VGG-19.

The forgery detection script effectively classifies unseen signatures based on the trained model.

🚀 Installation & Usage

🛠️ Requirements

Ensure you have the following dependencies installed:

pip install tensorflow keras numpy matplotlib

🏋️‍♂️ Training the Model

Run the following script to train the model:

python train.py

🕵️‍♂️ Running Forgery Detection

To check for forgery using the trained model:

python check_forgery.py --model model.h5 --test_dir path/to/test/data

🔮 Future Improvements

Implement a Siamese Network for better verification.

Experiment with other pre-trained models such as ResNet.

Improve dataset preprocessing techniques.

🤝 Contributing

Feel free to fork this repository and submit pull requests for improvements.

📜 License

This project is licensed under the MIT License.
