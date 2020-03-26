# -*- coding: utf-8 -*-
"""
Created on Sun Nov 24 01:40:19 2019

@author: Yohan
"""

from tensorflow.keras import backend as k
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, BatchNormalization 
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dropout , GlobalAveragePooling2D
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint , LearningRateScheduler
from sklearn.metrics import (confusion_matrix, log_loss, f1_score,average_precision_score,
balanced_accuracy_score,brier_score_loss,classification_report,fbeta_score,
hamming_loss,precision_recall_curve,roc_auc_score,recall_score,zero_one_loss,r2_score,roc_curve,precision_score)
from xlwt import Workbook
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model

batch_size = 64
epochs = 10

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
    
    
base_model = VGG16(include_top = False, input_shape = (150, 150, 3))
x = base_model.output
x = Flatten()(x)
x = Dense(6, activation='relu')(x)
x = Dense(12, activation='relu')(x)
x = Dense(12, activation='relu')(x)
x = Dense(6, activation='relu')(x)
x = BatchNormalization()(x)
predictions = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(loss='binary_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])

lr_reduce = ReduceLROnPlateau(monitor='val_accuracy', factor=0.1, min_delta=0.0001, patience=1, verbose=1)
filepath="transferlearning_weightsVGG16For2New.hdf5"
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

model.load_weights("transferlearning_weightsVGG16For2.hdf5")
model.evaluate_generator(test_set,
                         workers = 8,
                         max_queue_size = 1000)
y_predict = model.predict_generator(test_set,
                         workers = 8,
                         max_queue_size = 1000)
y_pred = [1 if i > 0.5 else 0 for i in y_predict]
y_true = [0 if i<234 else 1 for i in range(624)]

listData2 = listData()
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
listData2.listToExcel(cmScore, ps, rs, f1, aps, bas, bsl, fbs, hl, zol, rac)
listData2.wb1.save('VGG16For2.xls')

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



plot_model(model, to_file='VGG16For2.jpeg', dpi = 96)