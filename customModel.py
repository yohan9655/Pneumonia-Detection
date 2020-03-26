# -*- coding: utf-8 -*-
"""
Created on Thu Mar 26 14:36:48 2020

@author: Yohan
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 00:03:48 2020

@author: Yohan
"""
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.models import Model
import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py
import cv2
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
from tensorflow.keras.layers import Dropout , GlobalAveragePooling2D, BatchNormalization
from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint , LearningRateScheduler
from sklearn.metrics import (confusion_matrix, log_loss, f1_score,average_precision_score,
balanced_accuracy_score,brier_score_loss,classification_report,fbeta_score,
hamming_loss,precision_recall_curve,roc_auc_score,recall_score,zero_one_loss,r2_score,roc_curve,precision_score)
from xlwt import Workbook
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import numpy as np
        
class Excel:
    def __init__(self, wb, sheet):
        self.wb = wb
        self.i = 1
        self.sheet = sheet
        sheet.write(0, 0, 'InceptionFor2')
        sheet.write(1, 0,'Confusion Matrix')
        sheet.write(2, 0,'Precision Score')
        sheet.write(3, 0, 'Recall Score')
        sheet.write(4, 0, 'F1 Score')
        sheet.write(5,0,'Average Precision Score')
        sheet.write(6,0,'Balanced Accuracy Score')
        sheet.write(7,0,'Brier Score Loss')
        sheet.write(8,0,'Fbeta Score')
        sheet.write(9,0,'Hamming Loss')
        sheet.write(10,0,'Zero One Loss')
        sheet.write(11,0,'Area Under the Receiver Operating Characteristic Curve')
    
    def AddToExcel(self, listOut, col, row):
        #for i in range(len(listOut)-1):
        [self.sheet.write(row, col, listOut)]
        
class listData:
    def __init__(self):
        self.wb1 = Workbook()
        self.sheetWb1 = self.wb1.add_sheet('Sheet1', cell_overwrite_ok='False')
        self.excel1 = Excel(self.wb1, self.sheetWb1)
    def listToExcel(self, a, b, c, d, e, f, g, h, i, j, k):
    
        self.excel1.AddToExcel(a, 1, 1)
        
        self.excel1.AddToExcel(b, 1, 2)
        
        self.excel1.AddToExcel(c, 1, 3)
        
        self.excel1.AddToExcel(d, 1, 4)
        
        self.excel1.AddToExcel(e, 1, 5)
        
        self.excel1.AddToExcel(f, 1, 6)
        
        self.excel1.AddToExcel(g, 1, 7)
        
        self.excel1.AddToExcel(h, 1, 8)
        
        self.excel1.AddToExcel(i, 1, 9)
        
        self.excel1.AddToExcel(j, 1, 10)
    
        self.excel1.AddToExcel(k, 1, 11)
        
        return True

def layers(initial):
    rows = None
    cols = None
    
    #Batch Normalisation option
    
    batch_norm = 0
    kernel = (3, 3)
    layer1 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01))(initial)
    maxPool1 = tf.keras.layers.MaxPooling2D(strides=2)(layer1)
    layer3 = tf.keras.layers.Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same')(maxPool1)
    layer4a = tf.keras.layers.Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01))(layer3)
    layer4a= BatchNormalization()(layer4a)
    layer4b = tf.keras.layers.Conv2D(128,kernel_size = (4,4), activation = 'relu', padding='same')(layer3)
    layer5 = tf.keras.layers.concatenate([layer4a, layer4b], axis = 3)
    drop1 = tf.keras.layers.Dropout(0.5)(layer5)
    maxPool2 = tf.keras.layers.MaxPooling2D(strides=2)(layer5)
    layer6 = tf.keras.layers.Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same')(maxPool2)
    #layer7 = tf.keras.layers.Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same')(layer6)
    layer7 = BatchNormalization()(layer6)
    #maxPool3 = tf.keras.layers.MaxPooling2D(strides=2)(layer8)
    layer9 = tf.keras.layers.Conv2D(256, kernel_size = kernel,activation = 'relu', padding='same', kernel_regularizer=tf.keras.regularizers.l1(0.01), activity_regularizer=tf.keras.regularizers.l2(0.01))(layer7)
    drop1 = tf.keras.layers.Dropout(0.5)(layer9)
    layer10 = tf.keras.layers.Conv2D(256, kernel_size = kernel,activation = 'relu', padding='same')(layer9)
    return layer10
    
def CustomNet():  

    #Variable Input Size
    rows = None
    cols = None 
    kernel = (3, 3)


    input_flow = tf.keras.layers.Input(shape=(150, 150, 3))
    layer1 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(input_flow)
    """
    new1 = layers(layer1)
    new2 = layers(layer1)
    layer18 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(new1)
    layer19 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(new1)
    layer20 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(new1)
    layer21 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(new1)
    layer22 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(new2)
    layer23 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(new2)
    layer24 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(new2)
    layer25 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(new2)
    """
    layer18 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(layer1)
    layer19 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(layer1)
    new3a = layers(layer18)
    #new3b = layers(layer18)
    #finala = tf.keras.layers.concatenate([new3a,new3b], axis = 3)
    new3e = layers(layer19)
    #new3f = layers(layer19)
    #fih = layers(layer18)
    finalb = tf.keras.layers.concatenate([new3a,new3e], axis = 3)
    #maxPool4a = tf.keras.layers.MaxPooling2D(strides=2)(finala)
    maxPool4b = tf.keras.layers.MaxPooling2D(strides=2)(finalb)
    #finala = tf.keras.layers.Conv2D(256, kernel_size = kernel,activation = 'relu', padding='same')(maxPool4a)
    finalb = tf.keras.layers.Conv2D(256, kernel_size = kernel,activation = 'relu', padding='same')(maxPool4b)
    #concat12_21 = tf.keras.layers.concatenate([finala, finalb], axis = 3)
    
    #new3 = tf.keras.layers.Conv2D(1024, kernel_size = kernel,activation = 'relu', padding='same')(concat2_21)
    #concat1_2 = tf.keras.layers.concatenate([new1, new3], axis = 3)
    """
    layer2 = tf.keras.layers.Conv2D(64, kernel_size = kernel,activation = 'relu', padding='same')(layer1)
    maxPool1 = tf.keras.layers.MaxPooling2D(strides=2)(layer2)
    layer3 = tf.keras.layers.Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same')(maxPool1)
    layer4a = tf.keras.layers.Conv2D(128,kernel_size = kernel, activation = 'relu', padding='same')(layer3)
    layer4b = tf.keras.layers.Conv2D(128,kernel_size = (4,4), activation = 'relu', padding='same')(layer3)
    layer5 = tf.keras.layers.concatenate([layer4a, layer4b], axis = 3)
    maxPool2 = tf.keras.layers.MaxPooling2D(strides=2)(layer5)
    layer6 = tf.keras.layers.Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same')(maxPool2)
    layer7 = tf.keras.layers.Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same')(layer6)
    layer8 =  tf.keras.layers.Conv2D(256,kernel_size = kernel, activation = 'relu', padding='same')(layer7)
    maxPool3 = tf.keras.layers.MaxPooling2D(strides=2)(layer8)
    layer9 = tf.keras.layers.Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same')(maxPool3)
    layer10 = tf.keras.layers.Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same')(layer9)
    layer11 = tf.keras.layers.Conv2D(512, kernel_size = kernel,activation = 'relu', padding='same')(layer10)
    """
    

    
    #Conv2D
    """
    layer12 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, padding = 'same')(concat12_21)
    layer13 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, padding = 'same')(layer12)
    layer14 = tf.keras.layers.Conv2D(512, (3, 3), activation='relu', dilation_rate = 2, padding = 'same')(layer13)
    layer15 = tf.keras.layers.Conv2D(256, (3, 3), activation='relu', dilation_rate = 2, padding = 'same')(layer14)
    """
    layer16 = tf.keras.layers.Conv2D(128, (3, 3), activation='relu', dilation_rate = 2, padding = 'same')(finalb)
    layer17 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', dilation_rate = 2, padding = 'same')(layer16)
    final = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', dilation_rate = 1, padding = 'same')(layer17)

    sgd = SGD(lr = 1e-7, decay = (5*1e-4), momentum = 0.95)
    x = tf.keras.layers.Flatten()(final)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    x = tf.keras.layers.Dense(16, activation='relu')(x)
    x = BatchNormalization()(x)
    predictions = tf.keras.layers.Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_flow, outputs=predictions)
    model.compile(loss='binary_crossentropy', 
                      optimizer='adam', 
                      metrics=['accuracy'])		
    return model

batch_size = 16
epochs = 10

train_datagen = ImageDataGenerator()
    
test_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory('chest_xray/train',
                                                 target_size=(150,150),
                                                 batch_size=batch_size,
                        					     class_mode='binary') 
    
test_set = test_datagen.flow_from_directory('chest_xray/test',
                        					 target_size=(150,150),
                        					 batch_size=batch_size,
                        					 class_mode='binary',
                                             shuffle = False)

validation_set = test_datagen.flow_from_directory('chest_xray/val',
                                                  target_size=(150,150),
                                                  batch_size = batch_size,
                                                  class_mode='binary')



model = CustomNet()

import tensorflow as tf
tf.keras.utils.plot_model(model, show_shapes=True, to_file = 'modelA_1.jpg')

lr_reduce = ReduceLROnPlateau(monitor='val_acc', factor=0.1, min_delta=0.0001, patience=1, verbose=1)
#Reducing the learning rate timely using callbacks in Keras and also checkpointing the model when achieved the best so far quantity that is monitored , in this dataset I am monitoring validation accuracy.

#Saving the weights of the best model after checkpointing in transferlearning_weights.hdf5 .	         
filepath="transferlearning_weightsCustom1For2New.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

history = model.fit_generator(training_set,
    			         steps_per_epoch=5208/batch_size, #no. of images on training set/batch size
    			         epochs=10,
                         use_multiprocessing = False,
                         validation_data = test_set,
                         validation_steps = 624/batch_size,
                         workers = 8,
                         max_queue_size = 1000,
                         callbacks=[lr_reduce,checkpoint])



model.load_weights("transferlearning_weightsCustom1For2New.hdf5")
model.evaluate_generator(training_set,
                         workers = 8,
                         max_queue_size = 1000)

y_predict = model.predict_generator(test_set,
                         workers = 8,
                         max_queue_size = 1000)

y_pred = [1 if i > 0.5 else 0 for i in y_predict]
y_true = [0 if i<234 else 1 for i in range(624)]

listData1 = listData()
cm = confusion_matrix(y_true, y_pred)
cm5 = []
cm5 = confusion_matrix(y_pred, y_true)
cmScore = (cm5[0,0]+cm5[1,1])/(cm5[0,0] + cm5[0,1] + cm5[1,0] + cm5[1,1])
    #log = log_loss(y_pred, y_true)
ps = precision_score(y_pred, y_true)
rs = recall_score(y_pred, y_true)
f1 = f1_score(y_pred, y_true)
aps = average_precision_score(y_pred, y_true)
bas = balanced_accuracy_score(y_pred, y_true)
bsl = brier_score_loss(y_pred, y_true)
fbs = fbeta_score(y_pred, y_true, beta=1)
hl = hamming_loss(y_pred, y_true)
zol = zero_one_loss(y_pred, y_true)
rac = roc_auc_score(y_true, y_predict)
listData1.listToExcel(cmScore, ps, rs, f1, aps, bas, bsl, fbs, hl, zol, rac)
listData1.wb1.save('CustomModelFor2.xls')


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
plt.close()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plot_model(model, to_file='CustomNet.jpeg', dpi = 768)