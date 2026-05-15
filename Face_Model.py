#%%
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping

#conv2d- image ke features detect krta h
#MaxPooling2D- important features save krta h
#Flatten- image data ko 1D me convert krta h
#Dense- final prerdiction krta h

train_path = "train"
#train- isme sari image load h reaction ki

train_data = ImageDataGenerator(rescale=1./255)
#images ki pixel value ko normalize krta h
#normal - 0 to 255 (RGB)
#normalize ke bad - 0 to 1

train_dataset = train_data.flow_from_directory(
    train_path,
    target_size=(48,48),
    color_mode="grayscale",
    batch_size=32,
    class_mode="categorical"
)

model = Sequential()

model.add(Conv2D(32, (3,3), activation='relu', input_shape=(48,48,1))) 
model.add(MaxPooling2D(pool_size=(2,2))) 
# Second convolution layer.
model.add(Conv2D(64, (3,3), activation='relu')) #isme aur deep features collect krega
model.add(MaxPooling2D(pool_size=(2,2))) #Again omage size reduce krta h

model.add(Flatten())

model.add(Dense(128, activation='relu')) #128 Neurons ki quantity h hidden layers

model.add(Dense(7, activation='softmax')) #7 emotions detect krega 

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
#epochs dene se jyada overfitting bi ho jati h esliye hum log early stoping use krte h
#agr accuracy improve na ho to training automatically stop ho jaye
modelstop=EarlyStopping(
    monitor="accuracy",patience=2,
    restore_best_weights=True
)

model.fit(train_dataset, epochs=15) #datset 20 bar pura chalega

model.save("emotion_model.h5")

print("Model Trained Successfully")
# %%
loss, accuracy = model.evaluate(train_dataset)

print("Loss:", loss)
print("Accuracy:", accuracy)
# %%
