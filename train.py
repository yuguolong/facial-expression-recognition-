from tensorflow.keras.models import Model
from tensorflow.keras.applications.mobilenet import MobileNet
from tensorflow.keras.layers import Dense,GlobalAveragePooling2D,Dropout,LeakyReLU
from tensorflow import keras
from tensorflow.keras import regularizers
from keras.preprocessing.image import ImageDataGenerator

#model
base_model = MobileNet(alpha=0.5, depth_multiplier=1, dropout=1e-3, include_top=False)
# base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
x = Dense(128,activation=LeakyReLU(alpha=0.01))(x)
predictions = Dense(7, activation='softmax',activity_regularizer=regularizers.l1(0.01))(x)


model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['acc'])


#数据读入
train_datagen = ImageDataGenerator(rescale=1./255,
                                   horizontal_flip=True,
                                   rotation_range=3,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   zoom_range=0.2
                                   )
validation_datagen = ImageDataGenerator(rescale=1./255)
train_dir = './data/train'
validation_dir = './data/test'
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    batch_size=512,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(48, 48),
    batch_size=256,
    class_mode='categorical')



cp_callback = keras.callbacks.ModelCheckpoint('./logs/'+'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}-acc{acc:.3f}-val_acc{val_acc:.3f}.h5',
                                          save_best_only=True,save_weights_only=True,verbose=1)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss',
                                              factor=0.2,patience=5, min_lr=0.001) #loss不下降，降低学习率
reduce_tr = keras.callbacks.EarlyStopping(monitor='loss', patience=10, verbose=0, mode='auto')

model.fit(
    train_generator,
    epochs=40,
    validation_data=validation_generator,
    verbose=1,
    callbacks=[cp_callback,reduce_lr,reduce_tr])