import numpy as np
import tensorflow as tf
from tensorflow.keras import Input,Model
from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Layer
import matplotlib.pyplot as plt
# Avoid OOM errors by setting GPU Memory Consumption Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

#get images directories
anc_path="E:\\opencv\\Siamese\\anchor"
pos_path="E:\\opencv\\Siamese\\positive"
neg_path="E:\\opencv\\Siamese\\negtive"

anchor=tf.data.Dataset.list_files(anc_path+'\*.jpg')
positive=tf.data.Dataset.list_files(pos_path+'\*.jpg')
negative=tf.data.Dataset.list_files(neg_path+'\*.jpg')


#scale and resize
def preprocess(file_path):
    byte_img=tf.io.read_file(file_path)
    img=tf.io.decode_jpeg(byte_img)
    img=tf.image.resize(img,(100,100))
    img=img/255.0
    return img


#create dataset with labels
# 1:matched  0: not matched
positives=tf.data.Dataset.zip((anchor,positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
negatives=tf.data.Dataset.zip((anchor,negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
data=positives.concatenate(negatives)

#train and test partition
def preprocess_twin(input_img,val_img,label):
    return (preprocess(input_img),preprocess(val_img),label)

#build dataloader
data=data.map(preprocess_twin)
data=data.cache()
data=data.shuffle(buffer_size=1024)

#train data
train_data=data.take(round(len(data)*0.9))
train_data=train_data.batch(16)
train_data=train_data.prefetch(8)
xx=train_data.as_numpy_iterator().next()[0:2]
print(xx)
#test data
test_data=data.skip(round(len(data)*0.9))
test_data=test_data.take(round(len(data)*0.1)).batch(16).prefetch(8)


#embeding layer
def make_embeding():
    inp=Input(shape=(100,100,3),name='input_image')

    #first
    c1=Conv2D(64,(10,10),activation="relu")(inp)
    m1=MaxPool2D(64,(2,2),padding='same')(c1)

    #second
    c2=Conv2D(128,(7,7),activation='relu')(m1)
    m2=MaxPool2D(64,(2,2),padding='same')(c2)

    #third
    c3=Conv2D(128,(4,4),activation='relu')(m2)
    m3=MaxPool2D(64,(2,2),padding='same')(c3)

    #FINAL
    c4=Conv2D(256,(4,4),activation='relu')(m3)
    f1=Flatten()(c4)
    d1=Dense(4096,activation='sigmoid')(f1)

    return Model(inputs=[inp],outputs=[d1],name='embeding')

embeding=make_embeding()
#distance layer
class L1Dist(Layer):
    def __init__(self):
        super().__init__()
    def call(self,input_embedding,val_embedding):
        return tf.math.abs(input_embedding-val_embedding)
l1=L1Dist()

#Siamese neural network
def make_siamese_model():
    input_img=Input(name='input_img',shape=(100,100,3))
    val_img=Input(name='val_img',shape=(100,100,3))

    siamese_layer=L1Dist()
    siamese_layer._name='distance'
    distances=siamese_layer(embeding(input_img),embeding(val_img))

    classifier=Dense(1,activation='sigmoid')(distances)
    return Model(inputs=[input_img,val_img],outputs=classifier,name='siameseNetwork')

siamese_model=make_siamese_model()

#training
binary_cross_loss = tf.losses.BinaryCrossentropy()
opt = tf.keras.optimizers.Adam(1e-4) # 0.0001


class Siamese(Model):
    def __init__(self,siamese_model,**kwargs):
        super().__init__(**kwargs)
        self.model=siamese_model

    def compile(self, opt, classloss, **kwargs):
        super().compile(**kwargs)
        self.closs = classloss
        self.opt = opt

    def train_step(self,batch):
        # Record all of our operations
        with tf.GradientTape() as tape:
            # Get anchor and positive/negative image
            X = batch[:2]
            # Get label
            y = batch[2]

            # Forward pass
            yhat = self.model(X, training=True)
            # Calculate loss
            loss = self.closs(y, yhat)
        print(loss)

        # Calculate gradients
        grad = tape.gradient(loss, self.model.trainable_variables)

        # Calculate updated weights and apply to siamese model
        opt.apply_gradients(zip(grad, self.model.trainable_variables))

        # Return loss
        return  {"loss": loss}

    def test_step(self, batch, **kwargs):
        X = batch[:2]
        # Get label
        y = batch[2]

        yhat = self.model(X, training=False)
        loss=self.closs(y,yhat)

        return {"loss": loss}



    def call(self, X, **kwargs):
        return self.model(X, **kwargs)

model=Siamese(siamese_model)
model.compile(opt, binary_cross_loss)
model.compute_output_shape(input_shape=((None, 100, 100, 3),(None, 100, 100, 3)))
#train
checkpoint =tf.keras.callbacks.ModelCheckpoint(filepath='E:\\opencv\\Siamese\\siamese.h5py', monitor='val_loss', mode='min', save_best_only='True', verbose=1)
hist = model.fit(train_data, epochs=30, validation_data=test_data, callbacks=checkpoint)


#plot perfomance
fig, ax = plt.subplots(ncols=1, figsize=(120,50))

ax.plot(hist.history['loss'], color='teal', label='loss')
ax.plot(hist.history['val_loss'], color='orange', label='val loss')
ax.title.set_text('Loss')
ax.legend()

plt.savefig('history.jpg')





