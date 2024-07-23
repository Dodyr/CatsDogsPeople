import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
import numpy as np
import pickle
from tensorflow.keras.applications import MobileNet
from sklearn.linear_model import LogisticRegression

# Load the pre-trained model
with open('cat_or_dog.pckl', 'rb') as f:
    model = pickle.load(f)
    
base_model = MobileNet(
    include_top=False, # remove the top dense layers
    input_shape=(224,224,3),
    pooling='avg' # average pooling transforms 4d tensor to 2d feature matrix
)

# Initialize the main window
root = tk.Tk()
root.title("Cat or Dog or Human Classifier")

# Set up the label to display the image
label = Label(root)
label.pack()

# Function to preprocess the image
def image_to_array(fn):
    """Load image, cut to square shape and shrink to (224,224,3)"""
    im = Image.open(fn).convert('RGB')
    if im.size[0] > im.size[1]:
        # landscape format -> cut left and right
        xstart = (im.size[0] - im.size[1]) // 2
        im = im.crop((xstart, 0, xstart + im.size[1], im.size[1]))
    else:
        # portrait format -> cut up and down
        ystart = (im.size[1] - im.size[0]) // 2
        im = im.crop((0, ystart, im.size[0], ystart + im.size[0]))
    im = im.resize((224, 224))
    return np.array(im)

# Function to classify the image
def classify_image(image_path):
    img = image_to_array(image_path)
    x = np.vstack(img).reshape(1, 224, 224, 3)
    x = x / 255.0
    X = base_model.predict(x)
    pred = model.predict(X)
    prob = model.predict_proba(X)
    return pred[0], prob[0]

# Function to upload an image
def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load and display the image
        img = Image.open(file_path)
        img.thumbnail((300, 300))
        img = ImageTk.PhotoImage(img)
        label.configure(image=img)
        label.image = img

        # Classify the image
        pred, prob = classify_image(file_path)
        prob_cat, prob_dog, prob_hum = prob[0], prob[1], prob[2]
        result_text = f'This is a {pred}\nCat: {prob_cat:.2f}, Dog: {prob_dog:.2f}, Human: {prob_hum:.2f}'
        result_label.config(text=result_text)
        result_label.pack()

# Button to upload an image
upload_btn = tk.Button(root, text="Upload Image",
                       font=("Comic Sans MS", 30),
                             fg="green",
                             bg="black",
                             padx=10,
                             pady=10,
                             command=upload_image)
upload_btn.pack()

# Label to display the classification result
result_label = Label(root, text="Result",
                     font=("Comic Sans MS", 30),
                             fg="green",
                             bg="black",
                             padx=10,
                             pady=10,)
result_label.pack()

# Run the application
root.mainloop()

"""@Inproceedings (Conference){asirra-a-captcha-that-exploits-interest-aligned-manual-image-categorization,
author = {Elson, Jeremy and Douceur, John (JD) and Howell, Jon and Saul, Jared},
title = {Asirra: A CAPTCHA that Exploits Interest-Aligned Manual Image Categorization},
booktitle = {Proceedings of 14th ACM Conference on Computer and Communications Security (CCS)},
year = {2007},
month = {October},
publisher = {Association for Computing Machinery, Inc.},
url = {https://www.microsoft.com/en-us/research/publication/asirra-a-captcha-that-exploits-interest-aligned-manual-image-categorization/},
edition = {Proceedings of 14th ACM Conference on Computer and Communications Security (CCS)},
}"""

"""@inproceedings{liu2015faceattributes,
  title = {Deep Learning Face Attributes in the Wild},
  author = {Liu, Ziwei and Luo, Ping and Wang, Xiaogang and Tang, Xiaoou},
  booktitle = {Proceedings of International Conference on Computer Vision (ICCV)},
  month = {December},
  year = {2015} 
}"""