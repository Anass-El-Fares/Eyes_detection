# ------- Training CNN using Keras (Tensorflow) -------------
# ---------------- El Fares Anass ---------------------------
# ------------------ 30/12/2021 -----------------------------

from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dropout,Conv2D,Flatten,Dense, MaxPooling2D, BatchNormalization

def generator(dir, gen=image.ImageDataGenerator(rescale=1./255), shuffle=True,batch_size=1,target_size=(24,24),class_mode='categorical' ):
    return gen.flow_from_directory(dir,batch_size=batch_size,shuffle=shuffle,color_mode='grayscale',class_mode=class_mode,target_size=target_size)

BS = 32
TS =(24,24)
train_batch = generator('Dataset/train',shuffle=True, batch_size=BS,target_size=TS)
valid_batch = generator('Dataset/test',shuffle=True, batch_size=BS,target_size=TS)
SPE = len(train_batch.classes)//BS
VS = len(valid_batch.classes)//BS
print(SPE,VS)

model = Sequential([
    Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(24,24,1)),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(32,(3,3),activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(1,1)),
    Dropout(0.25),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(2, activation='softmax')])

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.fit_generator(train_batch, validation_data=valid_batch,epochs=20,steps_per_epoch=SPE ,validation_steps=VS)
model.save('models/CNN_eye.h5', overwrite=True)
