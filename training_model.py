from tensorflow import keras
import numpy as np
import warnings
warnings.filterwarnings('ignore')

main = np.load('data/data.npy')
labels = np.load('data/labels.npy')

print('x shape : ',main.shape)
print('y shape : ',labels.shape)

nclasses = len(np.unique(labels))

gesture_model = keras.models.Sequential([
    keras.layers.Input(shape=[42]),
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(100,activation='relu'),
    keras.layers.Dense(nclasses,activation='softmax')
])

gesture_model.compile(loss='sparse_categorical_crossentropy',optimizer='sgd',metrics=['accuracy'])

gesture_model.fit(main,labels,validation_split=.2,epochs=2)

gesture_model.save('gesture_model.h5')