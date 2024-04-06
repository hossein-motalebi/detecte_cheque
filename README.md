# detecte_cheque
Overview
This project introduces a robust system for analyzing and interpreting handwritten dates and amounts on cheques. Utilizing a blend of machine learning techniques, the system leverages both supervised and unsupervised learning, trained on the MNIST dataset, to achieve accurate recognition of handwritten characters.

Features
Handwritten date and amount recognition on cheques.
Integration of supervised and unsupervised machine learning methods.
Utilization of the MNIST dataset for training the recognition model.
Application of image processing techniques for enhancing character recognition accuracy.
Files
main.py: The main script that orchestrates the model training, image processing, and prediction tasks.
detecte_cheque.py: Contains the core functionality for cheque image processing, character segmentation, and machine learning model interaction.
How It Works
Image Processing: Cheque images are processed to identify and segment the date and amount sections.
Character Recognition: Each character is analyzed using a model trained on the MNIST dataset, employing both supervised and unsupervised learning techniques.
Prediction: The system combines the recognized characters to output the final interpreted date and amount.
Usage
To use this project, run main.py with Python 3. Ensure you have the necessary dependencies installed, including [list any specific libraries used, like NumPy, scikit-learn, etc.].

Contributing
Contributions to this project are welcome. Please feel free to fork the repository, make your changes, and submit a pull request.
