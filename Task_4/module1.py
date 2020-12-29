
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


IMG_SIZE= 96
BATCH_SIZE=512
EPOCHS=5
DIM=128
#AKTUELLES FILE

def main():
    
    num_train_samples = len(open("files/train_triplets.txt").readlines())
    dataset = tf.data.TextLineDataset("files/train_triplets.txt")
    dataset = dataset.map(
                       lambda line: get_dataset(line),
                        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    
    x = int(num_train_samples*0.8)
    train_dataset = dataset.take(x)
    val_dataset = dataset.skip(x)
    
    model = siamese_network()
    
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=triplet_loss)
    


    train_dataset = train_dataset.shuffle(1024, reshuffle_each_iteration=True).batch(BATCH_SIZE).repeat()
    val_dataset = val_dataset.batch(BATCH_SIZE)

    return
    model.fit(
        train_dataset,
        steps_per_epoch=int(np.ceil(x/BATCH_SIZE)),
         epochs=EPOCHS,
         validation_data=val_dataset,
         validation_steps=10
            )

    return
    test_dataset = tf.data.TextLineDataset("files/test_triplets.txt ")   
    test_dataset = test_dataset.map(
                    lambda line: get_test_dataset(line),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    num_test_samples = len(open("files/test_triplets.txt").readlines())
    inference_model = get_inference_model(model)
     
    test_dataset = test_dataset.batch(BATCH_SIZE)
    predictions = inference_model.predict(
        test_dataset,
        steps=int(num_test_samples/BATCH_SIZE),
        verbose=1)
    np.savetxt("predictions.txt", predictions, fmt="%i")
    


def load_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))/.255
   
    return img


def get_dataset(line):
    name = tf.strings.split(line)
    anc = load_image(tf.io.read_file('food/' + name[0] + '.jpg'))
    pos = load_image(tf.io.read_file('food/' + name[1] + '.jpg'))
    neg = load_image(tf.io.read_file('food/' + name[2] + '.jpg'))
    x=[anc, pos, neg]
    return x, 1
    
def get_test_dataset(line):
    name = tf.strings.split(line)
    anc = load_image(tf.io.read_file('food/' + name[0] + '.jpg'))
    pos = load_image(tf.io.read_file('food/' + name[1] + '.jpg'))
    neg = load_image(tf.io.read_file('food/' + name[2] + '.jpg'))
    
    return tf.stack([anc, pos, neg], axis=0)


def create_embedding(dim):

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(128,(7,7),padding='same',input_shape=(dim[0],dim[1],dim[2],),activation='relu',name='conv1'))
    model.add(tf.keras.layers.MaxPooling2D((2,2),(2,2),padding='same',name='pool1'))
    model.add(tf.keras.layers.Conv2D(256,(5,5),padding='same',activation='relu',name='conv2'))
    model.add(tf.keras.layers.MaxPooling2D((2,2),(2,2),padding='same',name='pool2'))
    model.add(tf.keras.layers.Flatten(name='flatten'))
    model.add(tf.keras.layers.Dense(DIM,name='embeddings'))
    
    return model

def siamese_network():

    anc_in = tf.keras.Input((IMG_SIZE,IMG_SIZE, 3))
    pos_in = tf.keras.Input((IMG_SIZE,IMG_SIZE, 3))
    neg_in = tf.keras.Input((IMG_SIZE,IMG_SIZE, 3))

    embedding_model = create_embedding([IMG_SIZE, IMG_SIZE, 3])
    embedding_model.summary()
    enc_anc = embedding_model(anc_in)
    enc_pos = embedding_model(pos_in)
    enc_neg = embedding_model(neg_in)

    out = tf.keras.layers.concatenate([enc_anc, enc_pos, enc_neg], axis=-1)
    inputs=tf.keras.layers.concatenate([anc_in, pos_in, neg_in], axis=-1)
    net = tf.keras.models.Model(inputs=inputs,
                    outputs=out)
    net.summary()
    return net

def triplet_loss(y_true, y_pred, alpha=0.2, emb_dim=DIM):
    anc = y_pred[:,:emb_dim],
    pos = y_pred[:,emb_dim:2*emb_dim]
    neg = y_pred[:,2*emb_dim]
    dp = tf.reduce_mean(tf.square(anc-pos), axis=1)
    dn = tf.reduce_mean(tf.square(anc-neg), axis=1)
    loss = tf.maximum(dp-dn+alpha, 0)
    return loss


if __name__ == '__main__':
    main()