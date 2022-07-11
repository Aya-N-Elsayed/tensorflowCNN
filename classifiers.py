import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import time
import os 
import struct

class classifiers():
  
    def __init__(self):
        print("CNN")

    def load_mnist(self,kind):
        """Load MNIST data from `path`"""
        labels_path = os.path.join( '%s-labels.idx1-ubyte' %(kind))
        images_path = os.path.join('%s-images.idx3-ubyte'%(kind))
        

        with open(labels_path, 'rb') as lbpath:
            magic, n = struct.unpack('>II',
                                    lbpath.read(8))
            labels = np.fromfile(lbpath,
                                dtype=np.uint8)

        with open(images_path, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack(">IIII",
                                                imgpath.read(16))
            images = np.fromfile(imgpath,
                                dtype=np.uint8).reshape(len(labels), 784)

        return images, labels



    def run_classifier(self):
        X_data, y_data = self.load_mnist( kind='train')
        print('Rows: %d,  Columns: %d' % (X_data.shape[0], X_data.shape[1]))
        X_test, y_test = self.load_mnist( kind='t10k')
        print('Rows: %d,  Columns: %d' % (X_test.shape[0], X_test.shape[1]))
        X_data, X_test = X_data[..., np.newaxis]/255.0, X_test[..., np.newaxis]/255.0
        X_data =X_data.reshape((-1, 28, 28, 1))
        X_test =X_test.reshape((-1, 28, 28, 1))
        #Separate validation dataset from training data
        BUFFER_SIZE = 10000
        BATCH_SIZE = 100
        NUM_EPOCHS = 5

        X_train, y_train = X_data[:50000,:], y_data[:50000]
        X_valid, y_valid = X_data[50000:,:], y_data[50000:]

        print('Training:   ', X_train.shape, y_train.shape)
        print('Validation: ', X_valid.shape, y_valid.shape)
        print('Test Set:   ', X_test.shape, y_test.shape)
        
        model = tf.keras.Sequential() 
        model.add(tf.keras.layers.Conv2D(filters=4, kernel_size=(3,3), strides=(1, 1),input_shape=(28,28,1),
                                                                padding='valid',data_format='channels_last', name='conv_1', activation='relu'))
        print("Conv_1") 
        #model.compute_output_shape(input_shape=(28, 28, 1))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='pool_1'))
        print("pool_1")
        #model.compute_output_shape(input_shape=(28, 28, 1))
        model.add(tf.keras.layers.Conv2D(filters=2, kernel_size=(3, 3), strides=(3, 3),
                                                               padding='valid',data_format='channels_last', name='conv_2', activation='relu'))
        print("Conv_2")
        #model.compute_output_shape(input_shape=(28, 28, 1))
        model.add(tf.keras.layers.MaxPool2D(pool_size=(4, 4), strides=(4, 4), name='pool_2'))
        print("pool_2")
        #model.compute_output_shape(input_shape=(28, 28, 1))
        model.add(tf.keras.layers.Flatten())
        print("Flatten")
        #model.compute_output_shape(input_shape=(28, 28, 1))
        model.add(tf.keras.layers.Dense(
        units=10, name='fc_1',activation='softmax'))
        tf.random.set_seed(1)
        #model.build(input_shape=(None, 28, 28, 1))
        print("fc_1")
        #model.compute_output_shape(input_shape=(28, 28, 1))
        #Compile and train the CNN model
        model.compile(optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy'])
        model.summary()
        startTime = time.time()
        history = model.fit(X_train,y_train,batch_size=BATCH_SIZE,epochs=NUM_EPOCHS,shuffle=True,validation_data=(X_valid, y_valid))
        endTime = time.time()
        print("Fitting time",endTime-startTime )
        #SEvaluate the model using testing data
        test_results = model.evaluate(X_test, y_test)
        print('\nTest Acc. {:.2f}%'.format(test_results[1]*100))
        # summarize history for accuracy
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()