#!/usr/bin/env python
# coding: utf-8

# ###### CREATION OF TWO CUSTOM LAYERS IN TENSORFLOW

# In[1]:


#powerful library for building and training machine learning models
import tensorflow as tf


# In[2]:


#define two custom layers, one for addition (AddLayer) and one for multiplication (MultiplyLayer).
#These layers perform basic mathematical operations on their inputs.

class CustomAddLayer(tf.keras.layers.Layer):
    def __init__(self, name='custom_add_layer', **kwargs):
        super(CustomAddLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        # No additional weights or trainable parameters needed for addition
        super(CustomAddLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Element-wise addition
        return tf.add(inputs[0], inputs[1])

    def get_config(self):
        config = super(CustomAddLayer, self).get_config()
        return config

class CustomMultiplyLayer(tf.keras.layers.Layer):
    def __init__(self, name='custom_multiply_layer', **kwargs):
        super(CustomMultiplyLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        # No additional weights or trainable parameters needed for multiplication
        super(CustomMultiplyLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # Element-wise multiplication
        return tf.multiply(inputs[0], inputs[1])

    def get_config(self):
        config = super(CustomMultiplyLayer, self).get_config()
        return config

# Example usage:
input_a = tf.keras.layers.Input(shape=(10,), name='input_a')
input_b = tf.keras.layers.Input(shape=(10,), name='input_b')

# Adding custom layers
add_result = CustomAddLayer(name='custom_add')([input_a, input_b])
multiply_result = CustomMultiplyLayer(name='custom_multiply')([input_a, input_b])

# Creating models
add_model = tf.keras.Model(inputs=[input_a, input_b], outputs=add_result, name='addition_model')
multiply_model = tf.keras.Model(inputs=[input_a, input_b], outputs=multiply_result, name='multiplication_model')

# Displaying model summaries
add_model.summary()
multiply_model.summary()


# ######  COMMBINATION OF TWO LAYERS IN A THIRD CUSTOM LAYER

# In[3]:


# Combination of  these two layers in a third custom layer. Concatenate them or multiply them
class CustomCombineLayer(tf.keras.layers.Layer):
    def __init__(self, name='custom_combine_layer', **kwargs):
        super(CustomCombineLayer, self).__init__(name=name, **kwargs)

    def build(self, input_shape):
        # No additional weights or trainable parameters needed for combination
        super(CustomCombineLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        add_output = inputs[0]
        multiply_output = inputs[1]

        # Concatenate the outputs along the last axis (axis=-1)
        combined_output = tf.concat([add_output, multiply_output], axis=-1)

        return combined_output

    def get_config(self):
        config = super(CustomCombineLayer, self).get_config()
        return config

# Example usage:
input_a = tf.keras.layers.Input(shape=(10,), name='input_a')
input_b = tf.keras.layers.Input(shape=(10,), name='input_b')

# Adding custom layers
add_result = CustomAddLayer(name='custom_add')([input_a, input_b])
multiply_result = CustomMultiplyLayer(name='custom_multiply')([input_a, input_b])

# Combine custom layers using the third custom layer
combined_result = CustomCombineLayer(name='custom_combine')([add_result, multiply_result])

# Creating model
combined_model = tf.keras.Model(inputs=[input_a, input_b], outputs=combined_result, name='combined_model')

# Displaying model summary
combined_model.summary()


# ###### CREATE A MODEL AND OBSERVE WORKING OF BATCH INFERENCE

# In[4]:


import numpy as np

# Generating random input data for batch inference
num_samples = 5
input_data_a = np.random.rand(num_samples, 10)
input_data_b = np.random.rand(num_samples, 10)

# Performing batch inference
predictions = combined_model.predict([input_data_a, input_data_b])

# Displaying predictions
print("Input Data A:")
print(input_data_a)
print("\nInput Data B:")
print(input_data_b)
print("\nPredictions:")
print(predictions)


# ###### SPLITTING OF INPUT IMAGE INTO 4*4 TILES

# In[5]:


# Example input image
input_image = tf.keras.layers.Input(shape=(None, None, 3))  # Assuming RGB image

# Define the size of the tiles
tile_size = (4, 4, 3)  # Height, Width, Channels

# Use tf.image.extract_patches to split the image into tiles
tiles = tf.image.extract_patches(
    input_image,
    sizes=[1, tile_size[0], tile_size[1], 1],  # Batch, Height, Width, Channels
    strides=[1, tile_size[0], tile_size[1], 1],  # Batch, Height, Width, Channels
    rates=[1, 1, 1, 1],  # Batch, Height, Width, Channels
    padding='VALID'
)

# Reshape the tiles to have the desired shape (4x4 tiles)
tiles = tf.reshape(tiles, (-1, tile_size[0], tile_size[1], tile_size[2]))

# Creating a model to visualize the result
model = tf.keras.Model(inputs=input_image, outputs=tiles)
model.summary()

# Example usage with a random image
random_image = tf.random.normal((1, 16, 16, 3))  # Assuming a 16x16 RGB image
result_tiles = model.predict(random_image)

# Displaying the result
print("Input Image:")
print(random_image.numpy().squeeze().astype(int))
print("\nResult Tiles:")
print(result_tiles.astype(int))


# ###### GRAPH DATA STRUCTURE

# In[6]:


import random

class Graph:
    def __init__(self):
        self.graph = {}

    def add_edge(self, start, end):
        if start not in self.graph:
            self.graph[start] = []
        self.graph[start].append(end)

    def generate_random_connections(self, nodes, num_connections):
        for _ in range(num_connections):
            start = random.choice(nodes)
            end = random.choice(nodes)
            self.add_edge(start, end)

    def display_graph(self):
        for node, neighbors in self.graph.items():
            print(f"{node} -> {', '.join(neighbors)}")

# Example usage
nodes = ['A', 'B', 'C', 'D', 'E']
num_connections = 7

graph = Graph()
graph.generate_random_connections(nodes, num_connections)

print("Random Graph:")
graph.display_graph()


# In[7]:


# Custom Node Class
class GraphNode:
    def __init__(self, name, model):
        self.name = name
        self.model = model

    def apply_inference_rule(self, input_data):
        # Convert input data to NumPy arrays
        input_a = np.array(input_data[0])
        input_b = np.array(input_data[1])

        # Apply inference rule based on the output
        output = self.model.predict([input_a, input_b])
        
        # Assuming output is a NumPy array, access the first element
        output_value = output[0]

        if output_value > 50:
            return "High_Output_Node"
        elif output_value < 50:
            return "Low_Output_Node"
        else:
            return "Equal_Output_Node"

# Example usage
nodes = ['A', 'B', 'C', 'D', 'E']

# Create a graph with random connections
graph = Graph()
graph.generate_random_connections(nodes, 7)

# Create a model using the CustomAddLayer
input_a = tf.keras.layers.Input(shape=(1,))
input_b = tf.keras.layers.Input(shape=(1,))
add_result = CustomAddLayer(name='custom_add')([input_a, input_b])
add_model = tf.keras.Model(inputs=[input_a, input_b], outputs=add_result)

# Create nodes with custom models
node_A = GraphNode("A", add_model)
node_B = GraphNode("B", add_model)
node_C = GraphNode("C", add_model)
node_D = GraphNode("D", add_model)
node_E = GraphNode("E", add_model)

# Simulate random input data for inference
random_input = random.randint(1, 100)

# Perform inference in Node A
next_node = node_A.apply_inference_rule([[random_input], [random_input]])

# Display the result
print(f"Initial Random Input: {random_input}")
print(f"Node A Output: {add_model.predict([np.array([[random_input]]), np.array([[random_input]])])[0][0]}")
print(f"Next Node: {next_node}")



# ###### OBJECT DETTECTION MODELS
import zipfile
import os

# Specify the path to the zip file
zip_file_path = 'Jellyfish_image_dataset.zip'

# Specify the directory where you want to extract the contents
extract_dir = 'Object_Detection'

# Create the extraction directory if it doesn't exist
os.makedirs(extract_dir, exist_ok=True)

# Extract the contents of the zip file
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# In[29]:


import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# In[40]:


import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Paths to your train and validation datasets
train_dir = 'Train'
validation_dir = 'Valid'
img_height, img_width = 224, 224
batch_size = 20

# Create image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load pre-trained MobileNetV2 model without top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model layers
base_model.trainable = False

# Create a new model on top
model1 = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model1.fit(
    train_generator,
    epochs=3,
    validation_data=validation_generator
)


# In[38]:


import tensorflow as tf
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# Paths to your train and validation datasets
train_dir = 'Train'
validation_dir = 'Valid'
img_height, img_width = 299, 299  # InceptionV3 requires input size of (299, 299)
batch_size = 25

# Create image data generators
train_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Load pre-trained InceptionV3 model without top classification layer
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the base model layers
base_model.trainable = False

# Create a new model on top
model2 = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_generator.class_indices), activation='softmax')
])

# Compile the model
model2.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
model2.fit(
    train_generator,
    epochs=3,
    validation_data=validation_generator
)


# ###### MobileNetV2 & InceptionV3
MobileNetV2 and InceptionV3 are both convolutional neural network (CNN) architectures designed for computer vision tasks, but they have differences in terms of architecture, speed, and accuracy. Here's a brief comparison:

Model Architecture:

MobileNetV2: MobileNetV2 is designed specifically for mobile and embedded vision applications. It uses depthwise separable convolutions to reduce the number of parameters and computations. 
It has a lightweight structure, making it suitable for real-time applications on devices with limited computational resources.

InceptionV3: InceptionV3, also known as GoogLeNetV3, is part of the Inception family of models developed by Google. It uses a more complex architecture with inception modules, which are designed to capture features at different scales using parallel convolutions with different kernel sizes. InceptionV3 typically has a larger number of parameters compared to MobileNetV2.

Speed and Computational Efficiency:

MobileNetV2: MobileNetV2 is known for its computational efficiency and speed. It achieves a good balance between accuracy and model size, making it suitable for real-time applications on resource-constrained devices.

InceptionV3: InceptionV3 is a deeper and more computationally intensive model compared to MobileNetV2. While it achieves higher accuracy, it may be slower and requires more computational resources.

Accuracy:

MobileNetV2: MobileNetV2 is designed to be a lightweight model, and its primary focus is on efficiency. 
While it may not achieve state-of-the-art accuracy on some tasks, it provides a good trade-off between speed and accuracy.

InceptionV3: InceptionV3 generally achieves higher accuracy on various computer vision tasks. 
It is suitable for scenarios where achieving the highest possible accuracy is more critical than computational efficiency.

Use Cases:

MobileNetV2: Ideal for mobile and edge devices where computational resources are limited, and real-time performance is crucial. Commonly used in applications such as image classification, object detection, and segmentation on mobile devices.

InceptionV3: Suitable for tasks where higher accuracy is required, and computational resources are less constrained. 
Often used in applications like image recognition, fine-grained classification, and image analysis.


Issues faced: While performing this section, I could use only certain models to train and validate. 
Also it is too slow to perform. So I couln't use more epochs and do hyperparameter tuning properly. consider the consequences while I'm solving out the tasks.
# ###### QUANTIZED MODEL FOR MobileNet V2

# In[46]:


# Convert the model to a quantized model 
converter = tf.lite.TFLiteConverter.from_keras_model(model1)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model1 = converter.convert()

# Save the quantized model to a file
with open('quantized_model.tflite', 'wb') as f:
    f.write(quantized_model1)


# In[51]:


import numpy as np
from PIL import Image
import os

# Paths to your test dataset
test_dir = 'Valid'  # Replace with the actual path to your test dataset

# Create an image data generator for the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=1,  # Set batch_size to 1 for inference on individual images
    class_mode='categorical',
    shuffle=False  # Ensure the order of predictions matches the order of images
)

# Load the quantized TFLite model
interpreter = tf.lite.Interpreter(model_content=quantized_model1)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Perform inference on each image in the test dataset
all_predictions = []

ground_truth_labels = test_generator.classes  # Ground truth labels (integer classes)

for i in range(len(test_generator.filenames)):
    # Load an image for inference
    image_path = os.path.join(test_dir, test_generator.filenames[i])
    image = Image.open(image_path).resize((img_width, img_height))
    image = np.array(image) / 255.0  # Normalize the image

    # Prepare the input data
    input_data = np.expand_dims(image, axis=0).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    all_predictions.append(predicted_class)

# Calculate accuracy
correct_predictions = np.sum(np.array(all_predictions) == ground_truth_labels)
total_samples = len(ground_truth_labels)
accuracy = correct_predictions / total_samples

# Print or use the accuracy as needed
print("Accuracy:", accuracy)


# ###### QUANTIZED MODEL FOR Inception V3

# In[58]:


# Convert the new model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model2)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
quantized_model2 = converter.convert()

# Save the quantized model to a file
with open('quantized_model2.tflite', 'wb') as f:
    f.write(quantized_model2)


# In[59]:


from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier

model_spec = model_spec.ImageModelSpec.from_keras_model(model2)
image_classifier.create(model_spec, train_data=train_generator, validation_data=validation_generator)
quantized_model2 = model_spec.create_tflite(quantization_config='dr')

# Save the quantized model to a file
with open('quantized_model2.tflite', 'wb') as f:
    f.write(quantized_model2)


# In[ ]:


import numpy as np
from PIL import Image
import os

# Paths to your test dataset
test_dir = 'Valid'  # Replace with the actual path to your test dataset

# Create an image data generator for the test dataset
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=1,  # Set batch_size to 1 for inference on individual images
    class_mode='categorical',
    shuffle=False  # Ensure the order of predictions matches the order of images
)

# Load the quantized TFLite model
interpreter = tf.lite.Interpreter(model_content=quantized_model2)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Perform inference on each image in the test dataset
all_predictions = []

ground_truth_labels = test_generator.classes  # Ground truth labels (integer classes)

for i in range(len(test_generator.filenames)):
    # Load an image for inference
    image_path = os.path.join(test_dir, test_generator.filenames[i])
    image = Image.open(image_path).resize((img_width, img_height))
    image = np.array(image) / 255.0  # Normalize the image

    # Prepare the input data
    input_data = np.expand_dims(image, axis=0).astype(input_details[0]['dtype'])
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    all_predictions.append(predicted_class)

# Calculate accuracy
correct_predictions = np.sum(np.array(all_predictions) == ground_truth_labels)
total_samples = len(ground_truth_labels)
accuracy = correct_predictions / total_samples

# Print or use the accuracy as needed
print("Accuracy:", accuracy)


# ###### DIFFERENCES
The key difference between quantized models and full precision (FP32) models lies in the representation of numerical values. Quantization is a technique used to reduce the memory and computational requirements of a model by representing weights and activations with fewer bits. FP32 models use 32-bit floating-point precision for numerical values, while quantized models use a lower bit precision.

The primary differences between quantized models and FP32 models:

1. Precision of Numerical Values:

FP32 Model (Full Precision):
Weights and activations are represented using 32-bit floating-point numbers.
High precision allows for a wide range of values and fine-grained representation of numerical information.
Requires more memory and computational resources.

Quantized Model:
Weights and activations are represented using a lower bit precision, typically 8-bit integers (INT8) or even lower.
Lower precision reduces memory requirements and speeds up inference but sacrifices some level of numerical precision.

2. Memory Requirements:

FP32 Model:
Requires more memory due to the larger size of 32-bit floating-point numbers.
Higher memory requirements can be a limiting factor, especially on resource-constrained devices.

Quantized Model:
Requires less memory as a result of using lower precision for weights and activations.
Well-suited for deployment on devices with limited memory.

3. Computational Efficiency:

FP32 Model:
More computationally intensive due to the higher precision calculations.
Slower inference on devices with limited computational power.

Quantized Model:
Faster inference due to reduced precision and lower computational requirements.
Well-suited for deployment on edge devices with constrained computational resources.

4. Deployment Considerations:

FP32 Model:
Often used during the training and development phase for maximum numerical precision and accuracy.
Larger model sizes may be a concern for deployment on devices with limited storage.

Quantized Model:
Commonly used for deployment on edge devices, mobile devices, or IoT devices where memory and computational resources are limited.
Reduced precision may lead to a slight drop in accuracy, but the trade-off is often acceptable for the benefits in terms of speed and resource efficiency.
# In[ ]:





# In[ ]:




