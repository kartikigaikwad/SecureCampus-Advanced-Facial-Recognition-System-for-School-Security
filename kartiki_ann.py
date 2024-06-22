
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split

dataset_dir = 'C:\\college\\dataset\\dataset'

# Initializing lists to store images and labels
images = []
labels = []

# mapping of class names to class IDs
class_mapping = {
    'happy': 0,
    'sad': 1,
    'surprise': 2
}

# Iterating through 'train' and 'test' subfolders
for split in ['train', 'test']:
    split_dir = os.path.join(dataset_dir, split)

    # Iterating through class folders ('happy', 'sad', 'surprise')
    for class_name in class_mapping.keys():
        class_dir = os.path.join(split_dir, class_name)

        # Iterating through image files in the class folder
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if image is not None:
                images.append(image)
                labels.append(class_mapping[class_name])

# Converting the 'images' and 'labels' lists to NumPy arrays
images = np.array(images)
labels = np.array(labels)


# Converting the list of grayscale images to a NumPy array
pixel_values = np.array(images)

# Verifying the shape of the pixel_values array ([num_images, height, width])
print("Shape of pixel_values:", pixel_values.shape)

# Initialize an empty list to store flattened feature vectors
feature_vectors = []

# Iterating through the grayscale images
for image in images:
    # Flatten each image into a 1D array
    flattened_image = image.flatten()
    feature_vectors.append(flattened_image)

# Converting the list of feature vectors to a NumPy array
feature_vectors = np.array(feature_vectors)


print("Shape of feature_vectors:", feature_vectors.shape)


class Perceptron:
    def __init__(self, num_inputs):
        # Initialize weights and bias to small random values
        self.weights = np.random.randn(num_inputs)
        self.bias = np.random.randn()
    
    def predict(self, inputs):
        # Compute the weighted sum of inputs and add bias
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        
        # Apply the activation function (e.g. threshold at 0)
        return 0 if weighted_sum <= 0 else 1
    
    def train(self, training_data, labels, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            for inputs, label in zip(training_data, labels):
                # Make a prediction using the current weights and bias
                prediction = self.predict(inputs)
                
                # Update weights and bias based on prediction error
                error = label - prediction
                self.weights += learning_rate * error * inputs
                self.bias += learning_rate * error

# Assuming feature_vectors and labels from previous steps
X_train, X_test, y_train, y_test = train_test_split(feature_vectors, labels, test_size=0.2, random_state=42)
# Initializing the Perceptron with the number of input neurons
num_inputs = X_train.shape[1]
perceptron = Perceptron(num_inputs)

# training hyperparameters
learning_rate = 0.01
num_epochs = 100

# training the Perceptron
perceptron.train(X_train, y_train, learning_rate, num_epochs)

def evaluate(model, test_data, labels):
    correct_predictions = 0
    total_predictions = len(test_data)
    
    for inputs, label in zip(test_data, labels):
        prediction = model.predict(inputs)
        if prediction == label:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

# Evaluating the Perceptron on the test data
test_accuracy = evaluate(perceptron, X_test, y_test)
print("Test Accuracy:", test_accuracy)

# creating and training the perceptron model
class PerceptronModel:
    def __init__(self, num_inputs, num_classes):
        # Initialize multiple perceptrons for each class
        self.perceptrons = [Perceptron(num_inputs) for _ in range(num_classes)]
    
    def predict(self, inputs):
        # Predict using all perceptrons and return the class with the highest score
        scores = [perceptron.predict(inputs) for perceptron in self.perceptrons]
        return np.argmax(scores)
    
    def train(self, training_data, labels, learning_rate, num_epochs):
        for epoch in range(num_epochs):
            for inputs, label in zip(training_data, labels):
                # Train each perceptron for the corresponding class
                for class_index, perceptron in enumerate(self.perceptrons):
                    target = 1 if class_index == label else 0
                    prediction = perceptron.predict(inputs)
                    error = target - prediction
                    perceptron.weights += learning_rate * error * inputs
                    perceptron.bias += learning_rate * error

# initialize and train the perceptron model
num_classes = len(class_mapping)
perceptron_model = PerceptronModel(num_inputs, num_classes)

# Define training hyperparameters
learning_rate = 0.01
num_epochs = 100

# Train the Perceptron Model
perceptron_model.train(X_train, y_train, learning_rate, num_epochs)

#Evaluate the Perceptron Model
def evaluate_model(model, test_data, labels):
    correct_predictions = 0
    total_predictions = len(test_data)
    
    for inputs, label in zip(test_data, labels):
        prediction = model.predict(inputs)
        if prediction == label:
            correct_predictions += 1
    
    accuracy = correct_predictions / total_predictions
    return accuracy

test_accuracy = evaluate_model(perceptron_model, X_test, y_test)
print("Test Accuracy of Perceptron Model:", test_accuracy)


#happyface1
#image to test
image_path = 'C://college//dataset//dataset//test//happy//PrivateTest_218533.jpg' 
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocessing the image
if input_image is not None:
    # Resizing
    input_image = cv2.resize(input_image, (48, 48)) 

    # Flatten the preprocessed image to match the feature vector format
    input_features = input_image.flatten()

    # Using the trained perceptron model to make predictions
    predicted_class = perceptron_model.predict(input_features)

    # Mapping the predicted class index to the corresponding facial expression label
    class_mapping_reverse = {v: k for k, v in class_mapping.items()}
    predicted_expression = class_mapping_reverse[predicted_class]


    cv2.imshow('Input Image', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # displaying the image and predicted expression
    print("Predicted Facial Expression:", predicted_expression)
else:
    print("Failed to load the input image.")

#sadface2
image_path = 'C:\\college\\dataset\\dataset\\test\\sad\\PrivateTest_568359.jpg' 
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess the image
if input_image is not None:
    # Resizing
    input_image = cv2.resize(input_image, (48, 48))  

    # Flatten the preprocessed image to match the feature vector format
    input_features = input_image.flatten()

    # Use the trained perceptron model to make predictions
    predicted_class = perceptron_model.predict(input_features)

    # Mapping the predicted class index to the corresponding facial expression label
    class_mapping_reverse = {v: k for k, v in class_mapping.items()}
    predicted_expression = class_mapping_reverse[predicted_class]
    
    cv2.imshow('Input Image', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    print("Predicted Facial Expression:", predicted_expression)
else:
    print("Failed to load the input image.")

#surprize face3
image_path = 'C:\\college\\dataset\\dataset\\test\\surprise\\PrivateTest_642696.jpg'  # Replace with the path to your image
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocessing the image
if input_image is not None:
    # resizing
    input_image = cv2.resize(input_image, (48, 48))  

    # Flattenning the preprocessed image to match the feature vector format
    input_features = input_image.flatten()

    # Using the trained perceptron model to make predictions
    predicted_class = perceptron_model.predict(input_features)

    # Mapping the predicted class index to the corresponding facial expression label
    class_mapping_reverse = {v: k for k, v in class_mapping.items()}
    predicted_expression = class_mapping_reverse[predicted_class]
    
    cv2.imshow('Input Image', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    print("Predicted Facial Expression:", predicted_expression)
else:
    print("Failed to load the input image.")

    #happyface2
image_path = 'C:\\college\\dataset\\dataset\\train\\happy\\Training_50580.jpg'
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocessing the image
if input_image is not None:
    # Resizing
    input_image = cv2.resize(input_image, (48, 48))  

    # Flattenning the preprocessed image to match the feature vector format
    input_features = input_image.flatten()

    # Using the trained perceptron model to make predictions
    predicted_class = perceptron_model.predict(input_features)

    # Mapping the predicted class index to the corresponding facial expression label
    class_mapping_reverse = {v: k for k, v in class_mapping.items()}
    predicted_expression = class_mapping_reverse[predicted_class]

    cv2.imshow('Input Image', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    print("Predicted Facial Expression:", predicted_expression)
else:
    print("Failed to load the input image.")

    #sadface
    #sadface2
image_path = 'C:\\college\\dataset\\dataset\\train\\sad\\Training_64796.jpg' 
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess the image
if input_image is not None:
    # Resizing
    input_image = cv2.resize(input_image, (48, 48))  

    # Flatten the preprocessed image to match the feature vector format
    input_features = input_image.flatten()

    # Use the trained perceptron model to make predictions
    predicted_class = perceptron_model.predict(input_features)

    # Mapping the predicted class index to the corresponding facial expression label
    class_mapping_reverse = {v: k for k, v in class_mapping.items()}
    predicted_expression = class_mapping_reverse[predicted_class]

    cv2.imshow('Input Image', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    print("Predicted Facial Expression:", predicted_expression)
else:
    print("Failed to load the input image.")

#surpriseface
image_path = 'C:\\college\\dataset\\dataset\\train\surprise\\Training_191269.jpg'  
input_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Preprocess the image
if input_image is not None:
    # Resizing
    input_image = cv2.resize(input_image, (48, 48))

    # Flatten the preprocessed image to match the feature vector format
    input_features = input_image.flatten()

    # Use the trained perceptron model to make predictions
    predicted_class = perceptron_model.predict(input_features)

    # Mapping the predicted class index to the corresponding facial expression label
    class_mapping_reverse = {v: k for k, v in class_mapping.items()}
    predicted_expression = class_mapping_reverse[predicted_class]

    cv2.imshow('Input Image', input_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    
    print("Predicted Facial Expression:", predicted_expression)
else:
    print("Failed to load the input image.")