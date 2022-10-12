import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
import keras
import numpy as np
from tensorflow.keras.preprocessing import image
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Flatten, Dropout

# load and preprocess imagenet images -- values in [0,1]
# images = load_preprocess_images(im_paths)
classes= ['Apple_Braeburn', 'Apricot', 'Avocado', 'Banana', 'Cherry', 'Guava', 'Lemon']

directory= os.getcwd()

def createTrainData():
    
    images_list = list()
    y=list()
    c=0
    for classname in classes:

        path=directory+"/mydata/Training/"+classname+"/"
        img_files= os.listdir(path)         #an array of the names of all files
        for i in img_files:
            finalpath= path+i 
            im = image.load_img(finalpath, target_size=(100,100,3))
            im = image.img_to_array(im)
            org_im_in = [(im/255).astype(np.float32)]
            images_list.append(np.array(org_im_in))
            y.append(c)
        c= c+1 
    return images_list, y

def createTestData(dir):

    images_list = list()
    y=list()
    c=0
    for classname in classes:
        path= directory+dir+classname+"/"
        img_files= os.listdir(path)         #an array of the names of all files
        for i in img_files:
            finalpath= path+i 
            im = image.load_img(finalpath, target_size=(100,100,3))
            im = image.img_to_array(im)
            org_im_in = [(im/255).astype(np.float32)]
            images_list.append(np.array(org_im_in))
            y.append(c)
        c= c+1 
    return images_list, y

def createModel():
    #classification model 
    model = Sequential() 
    model.add(Flatten(input_shape=x_train_array.shape[1:])) 
    model.add(Dense(100, activation=keras.layers.LeakyReLU(alpha=0.3))) 
    model.add(Dropout(0.5)) 
    model.add(Dense(50, activation=keras.layers.LeakyReLU(alpha=0.3))) 
    model.add(Dropout(0.3)) 
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['acc'])
    return model 


images_list, y= createTrainData()
x_train_array= np.array(images_list)
y_train_array= np.array(y)
#print(x_train_array.shape)
#print(len(images_list))
#exit()
#shuffle data
p = np.random.permutation(len(x_train_array))
x_train_array= x_train_array[p]     #traindata
y_train_array= y_train_array[p]
y_train= to_categorical(y_train_array)      #trainlabel


images_list, y= createTestData("/mydata/Test/")
x_test_array= np.array(images_list)
y_test_array= np.array(y)
p = np.random.permutation(len(x_test_array))
x_test_array= x_test_array[p]       #testdata
y_test_array= y_test_array[p]
y_test= to_categorical(y_test_array)        #testlabel

num_classes= 7

x_train_array= np.reshape(x_train_array,(x_train_array.shape[0],x_train_array.shape[2],x_train_array.shape[3],x_train_array.shape[4]))
x_test_array= np.reshape(x_test_array,(x_test_array.shape[0],x_test_array.shape[2],x_test_array.shape[3],x_test_array.shape[4]))

model= createModel() 

def fit_model_CNN(train_x, train_y, epoch_num):
    print(train_x.shape)
    #train_x=np.reshape(train_x, (train_x.shape[0], train_x.shape[1], 1))    #(batch_size, timestamps, features). Our usual 'features' are considered as 'timestamps' in LSTM. Each timestamp's (one cell in csv file) size is 1 (hence, features= 1) 
    model=createModel()
    model.fit(train_x, train_y, epochs=epoch_num, batch_size=64, verbose=2)
    return model

def trainModel():
    epoch_num = 100
    model = fit_model_CNN(x_train_array, y_train, epoch_num)
    filename = 'model_cnn' + '.h5'  # for CNN
    model.save(filename)
    print('> Saved model %s' % filename)

#Train the model
#trainModel()  
#print("TRAINING DONE!")

############################        T E S T     ###############################
print("LOADING PRETRAINED MODEL...")
mymodel= load_model("model_cnn.h5")
y_predicted= mymodel.predict(x_test_array)
acc = np.sum(np.argmax(y_predicted, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
print("Model Accuracy on Benign data:"+ str((acc * 100)))

####        TEST MODEL ON ADV DATA         ####

def test_on_pregenerated_adv_dataset(dir,e):
    images_list, y= createTestData(dir)
    x_test_array= np.array(images_list)
    y_test_array= np.array(y)
    p = np.random.permutation(len(x_test_array))
    x_test_array= x_test_array[p]       #testdata
    y_test_array= y_test_array[p]
    y_test= to_categorical(y_test_array)  
    y_predicted= mymodel.predict(x_test_array)      
    acc = np.sum(np.argmax(y_predicted, axis=1) == np.argmax(y_test, axis=1)) / y_test.shape[0]
    print("Model Accuracy on Adversarial(FGSM, e="+str(e)+") data: " +str((acc * 100)))


#test_on_pregenerated_adv_dataset("/mydata/TestAdv_fgsm_point_1/",0.1)
#test_on_pregenerated_adv_dataset("/mydata/TestAdv_fgsm_point_07/",0.07)
test_on_pregenerated_adv_dataset("/mydata/TestAdv_fgsm_point_17/",0.17)