# %% [markdown]
# ### resize image
# 

# %%
import os
import cv2
os.chdir("D:/Dataset/cancer")
img =cv2.imread('D:/Dataset/cancer/1.jpg')

# %%
import os
import cv2
os.chdir("D:/Dataset/Non-Cancer")
img =cv2.imread('D:/Dataset/Non-Cancer/1.jpg')

# %%
cv2.imshow('img',img)
cv2.waitKey(6000)
cv2.destroyAllWindows()

# %%
resize=cv2.resize(img,(256,256),interpolation=cv2.INTER_AREA)

# %%
cv2.imshow('res',resize)
cv2.waitKey(2000)
cv2.destroyAllWindows()

# %%
cv2.imwrite("img.jpg",resize)

# %%

for j in range(1,8):   
    for i in range(0,50):
        im=cv2.imread("D:/Dataset/cancer/"+str(j)+".jpg")
        resize=cv2.resize(im,(1028,1028),interpolation=cv2.INTER_AREA)
        cv2.imwrite("D:/Dataset/cancer/"+str(j)+str(i)+".jpg",resize)

# %%
for j in range(1,6):   
    for i in range(0,50):
        im=cv2.imread('D:/Dataset/Non-Cancer/'+str(j)+'.jpg')
        resize=cv2.resize(im,(1028,1028),interpolation=cv2.INTER_AREA)
        cv2.imwrite("D:/Dataset/Non-Cancer/"+str(j)+str(i)+".jpg",resize)

# %%
for i in range(1,8):
    n=str(i)
    img=cv2.imread('D:/dataset/cancer/'+n+'.jpg')
    resize=cv2.resize(img,(1028,1028),interpolation=cv2.INTER_AREA)
    cv2.imwrite("D:/Dataset/Resized cancer/"+n+".jpg",resize)

# %%
for i in range(1,6):
    n=str(i)
    img=cv2.imread('D:/dataset/Non-Cancer/'+n+'.jpg')
    resize=cv2.resize(img,(1028,1028),interpolation=cv2.INTER_AREA)
    cv2.imwrite("D:/Dataset/Resized Non-Cancer/"+n+".jpg",resize)

# %% [markdown]
# #Image Data Generation

# %%
from tensorflow.keras.utils import array_to_img, img_to_array, load_img

# %%
from keras.preprocessing.image import ImageDataGenerator

# %%
datagen = ImageDataGenerator(rotation_range=40,width_shift_range=0.1,
                            height_shift_range=0.1,shear_range=0.2,
                            zoom_range=0.2,horizontal_flip=False,
                            fill_mode='nearest')

# %%
def img_gene(img):
    img = load_img(img)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    i = 0
    for batch in datagen.flow(x, batch_size=1,save_to_dir='D:/Dataset/cancer',
                              save_prefix='10', save_format='jpg'):
        i+= 1
        if i > 20:
            break

# %%
for p in range(1,7):
    num=str(p)
    img="D:/Dataset/Resized cancer/"+num+".jpg"
    img_gene(img)

# %%
import os
# assign directory
directory = "D:/Dataset/Resized Non-Cancer/"
 
# iterate over files in
# that directory
for c in os.listdir(directory):
    f = os.path.join(directory, c)
    if os.path.isfile(f):
        #img="D:/Dataset/Resized cancer/"+num+".jpg"
        img_gene(f)

# %%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from time import perf_counter 
import os
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from tensorflow.keras.utils import plot_model

# %%
## Defining batch specfications
batch_size = 100
img_height = 250
img_width = 250

# %%
## loading training set
training_data = tf.keras.preprocessing.image_dataset_from_directory(
    'D:/Dataset/Train',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb'
)

# %%
## loading validation dataset
validation_data =  tf.keras.preprocessing.image_dataset_from_directory(
    'D:/Dataset/Val',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb'
)

# %%
## loading testing dataset
testing_data = tf.keras.preprocessing.image_dataset_from_directory(
    'D:/Dataset/test',
    seed=42,
    image_size= (img_height, img_width),
    batch_size=batch_size,
    color_mode='rgb'
)

# %%
testing_data

# %%
class_names = training_data.class_names
class_names

# %%
## Configuring dataset for performance
AUTOTUNE = tf.data.experimental.AUTOTUNE
training_data = training_data.cache().prefetch(buffer_size=AUTOTUNE)
testing_data = testing_data.cache().prefetch(buffer_size=AUTOTUNE)

# %%
## Defining Cnn
model = tf.keras.models.Sequential([
  layers.BatchNormalization(),
  layers.Conv2D(32, 3, activation='relu'), # Conv2D(f_size, filter_size, activation) # relu, sigmoid, softmax
  layers.MaxPooling2D(), # MaxPooling
  layers.Conv2D(64, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(128, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(256, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(512, activation='relu'),
  layers.Dense(len(class_names), activation= 'softmax')
])

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# %%
model.build((None, 250, 250, 3))
model.summary()

# %%
## lets train our CNN
checkpoint = ModelCheckpoint("model_weights.h5", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]
history = model.fit(training_data, validation_data=validation_data, epochs = 50, callbacks=callbacks_list)

# %%
###### serialize model structure to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# %%
## stats on training data
plt.plot(history.history['loss'], label = 'training loss')
plt.plot(history.history['accuracy'], label = 'training accuracy')
plt.grid(True)
plt.legend()

# %%
## stats on training data
plt.plot(history.history['val_loss'], label = 'validation loss')
plt.plot(history.history['val_accuracy'], label = 'validation accuracy')
plt.grid(True)
plt.legend()

# %%
## lets vizualize results on testing data
AccuracyVector = []
plt.figure(figsize=(30, 30))
for images, labels in testing_data.take(1):
    predictions = model.predict(images)
    predlabel = []
    prdlbl = []
    
    for mem in predictions:
        predlabel.append(class_names[np.argmax(mem)])
        prdlbl.append(np.argmax(mem))
    
    AccuracyVector = np.array(prdlbl) == labels
    for i in range(40):
        ax = plt.subplot(10, 4, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title('Pred: '+ predlabel[i]+' actl:'+class_names[labels[i]] )
        plt.axis('off')
        plt.grid(True)

# %%
from keras.models import model_from_json
import numpy as np

class cancerDetectionModel(object):

    class_nums = ['Training cancer', "Training Non-Cancer"]

    def __init__(self, model_json_file, model_weights_file):
        # load model from JSON file
        with open(model_json_file, "r") as json_file:
            loaded_model_json = json_file.read()
            self.loaded_model = model_from_json(loaded_model_json)

        # load weights into the new model
        self.loaded_model.load_weights(model_weights_file)
        self.loaded_model.make_predict_function()

    def predict_cancer(self, img):
        self.preds = self.loaded_model.predict(img)
        return cancerDetectionModel.class_nums[np.argmax(self.preds)], self.preds

# %%
full_train_df = pd.read_csv("D:/Dataset/converted_keras/labels.txt")
full_train_df.head()

# %%
print("Train Size: {}".format(len(os.listdir('D:\Dataset\Train\Training cancer'))))
print("Test Size: {}".format(len(os.listdir('D:\Dataset\Train\Training Non-Cancer'))))

# %%
import os
from PIL import Image
from matplotlib import pyplot as plt

# %%
root1 = 'D:\Dataset\Train\Training cancer'

fnames = os.listdir(root1)

# %%
len(fnames)

# %%
root2 = 'D:\Dataset\Train\Training Non-Cancer'

fnames = os.listdir(root2)

# %%
len(fnames)

# %%
fig , axs = plt.subplots(nrows=2,ncols=5,figsize=(10,10))
axs = axs.flatten()
for i in range(10):
    filepath = os.path.join(root1,root2,fnames[i])
    img = Image.open(filepath)
    axs[i].imshow(img)
    axs[i].axis('off')
    axs[i].set_title(fnames[i])
plt.show()

# %%
##### For CSV FORMATTTT
# from PIL import Image
# import numpy as np
# import sys
# import os
# import csv 

# def createFileList(myDir, format='jpg'):
#     fileList = []
#     print(myDir)
#     for root, dirs, files in os.walk(myDir, topdown = False):
#         for name in files:
#             if name.endswith(format):
#                 fullName = os.path.join(root, name)
#                 fileList.append(fullName)
#     return fileList
# myFileList = createFileList('D:\Dataset\Train\Training cancer')

# for file in myFileList:
#     print(file)
#     img_file = Image.open(file)
#     width, height = img_file.size
#     format = img_file.format
#     mode = img_file.mode
#     img_grey = img_file.convert('L')

#     value = np.asarray(img_grey.getdata(), dtype=np.int).reshape((img_grey.size[1], img_grey.size[0]))
#     value = value.flatten()
#     print(value)
#     with open("image_to_csv.csv", 'a') as f:
#         writer = csv.writer(f)
#         writer.writerow(value)



# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# %%
import os 
path = os.listdir('D:/Dataset/Train')
classes = {'Training cancer': 0, 'Training Non-Cancer': 1}

# %%
import cv2 
X = []
Y = []
for cls in classes:
    pth = 'D:/Dataset/Train/'+cls
    for j in os.listdir(pth):
        img = cv2.imread(pth+'/'+j,0)
        img = cv2.resize(img, (1028,1028))
        X.append(img)
        Y.append(classes[cls])



# %%
np.unique(Y)

# %%
X = np.array(X)
Y = np.array(Y)

# %%
pd.Series(Y).value_counts()

# %%
X.shape

# %%
#Visualize Data 
plt.imshow(X[0], cmap='gray')

# %%
#Prepare Data
X_updated = X.reshape(len(X), -1)
X_updated.shape

# %%
#split Data
xtrain, xtest, ytrain, ytest = train_test_split(X_updated, Y, random_state=10,test_size=.20)

# %%
xtrain.shape, xtest.shape

# %%
#Feature Scaling 
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())
xtrain = xtrain/255
xtest = xtest/255
print(xtrain.max(), xtrain.min())
print(xtest.max(), xtest.min())

# %%
#Fearute selection: PCA
from sklearn.decomposition import PCA

# %%
print(xtrain.shape, xtest.shape)
pca = PCA(.98)
# pca_train = pca.fit_transform(xtrain)
# pca_test = pca.transform(xtest)
pca_train = xtrain
pca_test = xtest

# %%
# print(pca_train.shape, pca_test.shape)
# print(pca.n_components_)
# print(pca.n_features_)

# %%
###TRAIN MODEL
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# %%
import warnings
warnings.filterwarnings('ignore')

lg = LogisticRegression(C=0.1) #Penality Parameter
lg.fit(pca_train, ytrain)


# %%
sv = SVC()
sv.fit(pca_train, ytrain)

# %%
#Evaluation
print("Training Score :", lg.score(pca_train, ytrain))
print("Testing Score:", lg.score(pca_test, ytest))

# %%
print("Training Score :", sv.score(pca_train, ytrain))
print("Testing Score:", sv.score(pca_test, ytest))

# %%
#Prediction
pred = sv.predict(pca_test)
np.where(ytest!=pred)

# %%
pred[36]

# %%
ytest[36]

# %%
#TEST MODEL
dec = {0:'cancer', 1:'Non-Cancer'}

# %%
plt.figure(figsize=(12,8))
p = os.listdir('D:/Dataset/test/')
c=1
for i in os.listdir('D:/Dataset/test/cancer/')[:8]:
    plt.subplot(3,3,c)

    img = cv2.imread('D:/Dataset/test/cancer/'+i,0)
    img1 = cv2.resize(img, (1028,1028))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1

# %%
plt.figure(figsize=(12,8))
p = os.listdir('D:/Dataset/test/')
c=1
for i in os.listdir('D:/Dataset/test/Non-Cancer/')[:6]:
    plt.subplot(3,3,c)

    img = cv2.imread('D:/Dataset/test/Non-Cancer/'+i,0)
    img1 = cv2.resize(img, (1028,1028))
    img1 = img1.reshape(1,-1)/255
    p = sv.predict(img1)
    plt.title(dec[p[0]])
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    c+=1

# %%
BASE_PATH = 'D:/Dataset/Train/'
unique_classes = []
for path in os.listdir(BASE_PATH):
    unique_classes.append(path)
print(unique_classes)

# %%
class_index = [unique_classes[1], unique_classes[0]]
for c in class_index:
    print(c, "-", class_index.index(c))

# %%
images = []
masks = []
labels = []
for folder in os.listdir(BASE_PATH):
    class_path = os.path.join(BASE_PATH, folder)
    for img in os.listdir(class_path):
        if "_mask" not in img:
            img_path = os.path.join(class_path, img)
            msk_path = img_path.replace(".png", "_mask.png")
            # check if mask exist
            if os.path.exists(msk_path):
                images.append(img_path)
                masks.append(msk_path)
                labels.append(folder)

# %%
print(len(images))

# %%
images[0]

# %%
input_images_size = 256
channel = 1

# %%
import cv2
import scipy
import scipy.ndimage


def load_image(img_path):
    """ Load single image as Grayscale
    """
    # load image as grayscale
    img = cv2.imread(img_path, 0)
    return img

def padding(img, msk):
    """ Pad images to make them square
    """
    size = np.max(img.shape)

    offset_x = (size-img.shape[0])//2
    offset_y = (size-img.shape[1])//2

    blank_image = np.zeros((size, size))
    blank_mask = np.zeros((size, size))

    blank_image[offset_x:offset_x+img.shape[0],
               offset_y:offset_y+img.shape[1]] = img
    blank_mask[offset_x:offset_x+img.shape[0],
               offset_y:offset_y+img.shape[1]] = msk
    return blank_image, blank_mask

def resize_mask(mask):
    """Resize mask, its different because mask pixel value can change because of resize
    """
    new_size = np.array([input_images_size, input_images_size]) / mask.shape
    mask = scipy.ndimage.interpolation.zoom(mask, new_size)
    return mask

def resize(img):
    """Resize image
    """
    img = cv2.resize(img, (input_images_size, input_images_size))
    return img
        
def preprocess(img):
    """Image preprocessing
    Normalize image
    """
    img = img/255.0
    return img

def inverse_preprocess(img):
    """Inverse of preprocessing
    """
    img = img*255
    return img

def load_data(img_path, msk_path, label):
    """Load image, mask and repalce mask value with class index
    0 = normal
    1 = benign
    2 = malignant
    """
    img = load_image(img_path)
    msk = load_image(msk_path)
    img, msk = padding(img, msk)
    label_indx = class_index.index(label)
    msk[msk == 255] = 1
    msk = msk.astype("uint8")
    img = resize(img)
    msk = resize_mask(msk)
    new_mask = np.zeros((input_images_size, input_images_size, 2))
    if label_indx != 0:
        new_mask[:, :, label_indx-1] = msk
#     print(np.unique(msk), label, label_indx)
    return img, new_mask

def load_batch(images, masks, labels):
    """Load Batch of data
    """
    batch_x = []
    batch_y = []
    for i, m, l in zip(images, masks, labels):
        img, msk = load_data(i, m, l)
        img = preprocess(img)
        batch_x.append(img)
        batch_y.append(msk)
    return np.array(batch_x), np.array(batch_y) 

# %%
import matplotlib.pyplot as plt
for i in [0, 500, 600]:
    indx = i
    img, msk = load_data(images[indx], masks[indx], labels[indx])
    print(np.min(img), np.max(img), img.shape)
    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(msk[:, :, 0])
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(msk[:, :, 1])
    plt.show()


# %%
images = np.array(images)
masks = np.array(masks)
labels = np.array(labels)

# %%
!pip install segmentation_models

# %%
pip install upgrade keras

# %%
import os
os.environ["SM_FRAMEWORK"] = "tf.keras"
from tensorflow import keras 
import segmentation_models as sm
import tensorflow as tf

sm.framework()

BACKBONE = 'resnet34'
LR = 0.00001
model = sm.Unet(BACKBONE, classes=2, activation="sigmoid",input_shape=(input_images_size,input_images_size, channel),encoder_weights=None)

optim = tf.keras.optimizers.Adam(LR)

dice_loss = sm.losses.DiceLoss()
focal_loss = sm.losses.BinaryFocalLoss()
total_loss = dice_loss + (1 * focal_loss)

metrics = [sm.metrics.IOUScore(threshold=0.5),
           sm.metrics.FScore(threshold=0.5)]

model.compile(optim, total_loss, metrics)

# %%
batch_size = 4
history = {"epoch": []}
for e in range(100):
    print("epoch:",e, end=" > ")
    indexes = list(range(len(images)))
    temp_history = {"loss": [],
                   "IOU": [],
                   "F-Score": []}
    for b in range(0, len(images), batch_size):
        bs = b
        be = bs+batch_size
        batch_index = indexes[bs:be]
        batch_x, batch_y = load_batch(images[batch_index], masks[batch_index], labels[batch_index])
        batch_x = np.expand_dims(batch_x, axis=-1)
        batch_y = np.expand_dims(batch_y, axis=-1)
        batch_y = batch_y.astype("float32")
        loss = model.train_on_batch(batch_x, batch_y)
        temp_history["loss"].append(loss[0])
        temp_history["IOU"].append(loss[1])
        temp_history["F-Score"].append(loss[2])
    print("loss", np.round(np.mean(temp_history["loss"]), 4),"IOU", np.round(np.mean(temp_history["IOU"]), 4),"F-Score", np.round(np.mean(temp_history["F-Score"]), 4))
    
    history["epoch"].append(temp_history)
    
model.save_weights("cancer")

# %%
import matplotlib.pyplot as plt
for i in [0, 500, 600]:
    indx = i
    img, msk = load_data(images[indx], masks[indx], labels[indx])
    print(np.min(img), np.max(img), img.shape)
    print(img.shape)
    
    img2 = preprocess(img)
    pred = model.predict(np.array([img2]))
    pred = pred[0]

    plt.figure(figsize=(20, 20))
    plt.subplot(1, 3, 1)
    plt.axis("off")
    plt.imshow(img)
    plt.subplot(1, 3, 2)
    plt.axis("off")
    plt.imshow(pred[:, :, 0])
    plt.subplot(1, 3, 3)
    plt.axis("off")
    plt.imshow(pred[:, :, 1])
    plt.show()

# %%
from keras.models import load_model  # TensorFlow is required for Keras to work
from PIL import Image, ImageOps  # Install pillow instead of PIL
import numpy as np

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("D:/Dataset/converted_keras/keras_model.h5", compile=False)

# Load the labels
class_names = open("D:/Dataset/converted_keras/labels.txt", "r").readlines()

# Create the array of the right shape to feed into the keras model
# The 'length' or number of images you can put into the array is
# determined by the first position in the shape tuple, in this case 1
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

# Replace this with the path to your image ... PATH...PATH..PATH
image = Image.open("D:/Dataset/test/Non-Cancer/6.png").convert("RGB")

# resizing the image to be at least 224x224 and then cropping from the center
size = (224, 224)
image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)

# turn the image into a numpy array
image_array = np.asarray(image)

# Normalize the image
normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1

# Load the image into the array
data[0] = normalized_image_array

# Predicts the model
prediction = model.predict(data)
index = np.argmax(prediction)
class_name = class_names[index]
confidence_score = prediction[0][index]

# Print prediction and confidence score
print("Class:", class_name[2:], end="")
print("Confidence Score:", confidence_score)



