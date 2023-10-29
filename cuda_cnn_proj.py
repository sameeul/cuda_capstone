
import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import matplotlib.pyplot as plt
from keras import utils 
from dataclasses import dataclass

@dataclass
class ImageData:
  dataset_name : str
  num_channel : int
  channel_id : int
  data_path : str

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def load_image_data(image_data_set):
    path = image_data_set.data_path 
    groups = ["G2","notG2"] # class name
    title = image_data_set.dataset_name + "_Results" # 
    num_ch = image_data_set.num_channel
    channel_id = 0 # default
    
    if num_ch == 1:
        channel_id = image_data_set.channel_id
    
    if(path[-1]=="/"):
        path=path[:-1]
    
    nb_classes = len(groups)

    x_train = np.load(path+"/data/x_train.npy")
    y_train = np.load(path+"/data/y_train.npy")

    x_test = np.load(path +"/data/X_test.npy")
    y_test = np.load(path +"/data/Y_test.npy")

    x_train = x_train/255
    x_test = x_test/255

    y_train = utils.to_categorical(y_train,nb_classes)
    y_test = utils.to_categorical(y_test,nb_classes)
    
    # shuffle sequence of data
    index=list(range(x_train.shape[0]))
    index=random.sample(index,len(index))
    x_train = x_train[index]
    y_train = y_train[index]

    if num_ch == 2:
        x_train=x_train[...,:2]
        x_test=x_test[...,:2]
    else:
        x_train=x_train[...,channel_id]
        x_test=x_test[...,channel_id]
        x_train=x_train[...,None]
        x_test=x_test[...,None]
        
    return x_train, y_train, x_test, y_test

def build_model(input_image_shape):
    model = keras.models.Sequential()
    model.add(layers.Conv2D(128, (3,3), strides=(1,1), padding="valid", activation='relu', input_shape=input_image_shape))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Conv2D(128, 3, activation='relu'))
    model.add(layers.MaxPool2D((2,2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(0.03))
    model.add(layers.Dense(2))
    
    loss = keras.losses.BinaryCrossentropy(from_logits=True)
    optim = keras.optimizers.Adam(learning_rate=0.001)
    metrics = ["accuracy"]

    model.compile(optimizer=optim, loss=loss, metrics=metrics)

    return model

def generate_plots(hist_data, dataset_name):
    
    fig, (axL, axR) = plt.subplots(ncols=2,figsize=(16,8))

    acc = hist_data.history['accuracy'] # accuracy for training data
    val_acc = hist_data.history['val_accuracy'] # accuracy for validation data
    loss = hist_data.history['loss'] # loss for training data
    val_loss = hist_data.history['val_loss'] # loss for validation
    
    plt.rcParams["font.size"] = 16
    
    axL.set_ylim([0,1.0])
    axL.plot(acc,label='Training acc')
    axL.plot(val_acc,label='Validation acc')
    axL.set_title(dataset_name + ' Accuracy')
    axL.legend(loc='best') # position of legend
    
    axR.set_ylim([0,1.0])
    axR.plot(loss,label='Training loss')
    axR.plot(val_loss,label='Validation loss')
    axR.set_title(dataset_name + ' Loss')
    axR.legend(loc='best')
    
    plt.savefig(f"{dataset_name}_training_history.png")
    

def run_CNN(image_data_set, batch_size, epochs):
    
    print(f"Running CNN for {image_data_set.dataset_name}")
    
    # load data
    x_train, y_train, x_test, y_test = load_image_data(image_data_set)
    print(f"Training dataset image shape {x_train.shape}")
    print(f"Training dataset label shape {y_train.shape}")
    print(f"Test dataset image shape {x_test.shape}")
    print(f"Test dataset label shape {y_test.shape}")
    
    # build model
    input_shape = x_train.shape[1:]
    model = build_model(input_shape)
    print(model.summary())
    
    #train model
    hist = model.fit(x_train, y_train,  validation_split=0.2, epochs=epochs, batch_size=batch_size, verbose=2)
    
    # save history plots
    generate_plots(hist, image_data_set.dataset_name)

    # evaulate model
    model.evaluate(x_test,  y_test, batch_size=batch_size, verbose=2)
    
if __name__ == "__main__":
    
    batch_size = 64
    epochs = 20
    
    Hoechst_image_data = ImageData("Hoechst", 1, 0, "/home/jovyan/work/cuda_capstone/HeLa_Hoechst-EB1/")
    run_CNN(Hoechst_image_data, batch_size, epochs)
    
    EB1_image_data = ImageData("EB1", 1, 1, "/home/jovyan/work/cuda_capstone/HeLa_Hoechst-EB1/")
    run_CNN(EB1_image_data, batch_size, epochs)
    
    Hoechst_EB1_image_data = ImageData("Hoechst-EB1", 2, 0, "/home/jovyan/work/cuda_capstone/HeLa_Hoechst-EB1/")
    run_CNN(Hoechst_EB1_image_data, batch_size, epochs)
    
    GM130_image_data = ImageData("GM130", 1, 1, "/home/jovyan/work/cuda_capstone/HeLa_Hoechst-GM130/")
    run_CNN(GM130_image_data, batch_size, epochs)
        
    Hoechst_GM130_image_data = ImageData("Hoechst-GM130", 2, 0, "/home/jovyan/work/cuda_capstone/HeLa_Hoechst-GM130/")
    run_CNN(Hoechst_GM130_image_data, batch_size, epochs)