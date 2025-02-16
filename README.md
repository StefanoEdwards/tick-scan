# tick-scan
Overview
TickScan is a web application designed to help users determine whether a bite is from a tick or a mosquito. This distinction is important as tick bites can lead to serious illnesses like Lyme disease, while mosquito bites are usually harmless. The app allows users to upload an image of a bite and receive an instant classification result. Users upload an image, which is then preprocessed and resized for analysis. A deep learning model, trained using Teachable Machine, classifies the image as either a tick or mosquito bite, and the app displays the predicted class along with a confidence score.

Frameworks Used
This project was built using Streamlit for the web interface, Keras and TensorFlow for the deep learning model, and Teachable Machine for training the classifier. Pillow (PIL) was used for image processing, while NumPy helped with numerical data handling.

Getting Started
To use TickScan, install the necessary dependencies by running pip install streamlit keras tensorflow pillow numpy. Then, launch the application using streamlit run app.py. Once the app is running, users can upload an image of a bug bite and receive an instant classification result.

Acknowledgments
This project was inspired by the need for early tick bite detection to prevent Lyme disease. To build it, I followed a tutorial by Computer Vision Engineer (aka Felipe), which taught me how to use Streamlit and Teachable Machine. The tutorial provided a foundation for training the model and developing the app's interface. Streamlit made it easy to create a user-friendly experience, and Teachable Machine allowed me to fine-tune the classifier without needing extensive machine learning expertise.

About the Author
I'm Stefano, a grade 10 student from Toronto with a passion for AI and computer vision. Feel free to reach out with any questions at: stefanoedwards@icloud.
