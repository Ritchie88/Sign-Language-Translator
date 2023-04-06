import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Dropout,BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

EPOCHES = 50
#The folders which Sign Language MNIST data is found
train = pd.read_csv("sign_mnist_train.csv")
test = pd.read_csv("sign_mnist_test.csv")

#Prepare the data to be used in the model, training and testing data
#Changes the data to 3-D
def PrepareData(train,test):
        trainData = train.values
        testData = test.values

        trainData = trainData / 255
        testData=testData / 255

        trainData = trainData.reshape(-1,28,28,1)
        testData = testData.reshape(-1,28,28,1)
        return trainData,testData 

#Constructs the model using keras Sequential model system, then compiles and runs model
def buildModel():
        model = Sequential()
        model.add(Conv2D(75 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu' , input_shape = (28,28,1)))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(50 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(Dropout(0.2))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Conv2D(25 , (3,3) , strides = 1 , padding = 'same' , activation = 'relu'))
        model.add(BatchNormalization())
        model.add(MaxPool2D((2,2) , strides = 2 , padding = 'same'))
        model.add(Flatten())
        model.add(Dense(units = 512 , activation = 'relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units = 24 , activation = 'softmax'))
        return model
        

def main():
        #colect the labels of each test/training sample, then remove them to have raw data
        yTrain = train['label']
        yTest = test['label']
        del train['label']
        del test['label']

        xTrain, xTest = PrepareData(train,test)

        from sklearn.preprocessing import LabelBinarizer
        LB = LabelBinarizer()
        yTrain = LB.fit_transform(yTrain)
        yTest = LB.fit_transform(yTest)

        datagen = ImageDataGenerator(
                featurewise_center=False,  # set input mean to 0 over the dataset
                samplewise_center=False,  # set each sample mean to 0
                featurewise_std_normalization=False,  # divide inputs by std of the dataset
                samplewise_std_normalization=False,  # divide each input by its std
                zca_whitening=False,  # apply ZCA whitening
                rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
                zoom_range = 0.1, # Randomly zoom image 
                width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                horizontal_flip=False,  # randomly flip images
                vertical_flip=False)  # randomly flip images

        #Set the sample data to be used in the Data Generator
        datagen.fit(xTrain)

        #Set the Callback Function
        learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy', patience = 2, verbose=1,factor=0.5, min_lr=0.00001)

        #Build the Model
        model = buildModel()
        #Compile the model, with a focus of an adam optimization of the cost function 
        model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        model.summary()

        #Run the model to the specified number of epochs
        #Use the training data for the training, Testing data for the validation
        #Callback is set to Reduce the learning rate when there is no significant change
        model.fit(datagen.flow(xTrain,yTrain, batch_size = 128) ,EPOCHES = 20 , validation_data = (xTest, yTest) , callbacks = [learning_rate_reduction])

        (ignore,res)=model.evaluate(x=xTest,y=yTest)
        print('MODEL ACCURACY = {}%'.format(res*100))
        model.save('SLmodel.h5')

if __name__ == '__main__':
    main()