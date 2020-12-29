
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMG_SIZE= 96
BATCH_SIZE=32
EPOCHS=10
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
    
    model = get_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=triplet_loss,
                  metrics=[accuracy])
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True)
    
    train_dataset = dataset.take(x)
    val_dataset = dataset.skip(x)
    train_dataset = train_dataset.batch(BATCH_SIZE).repeat()
    val_dataset = val_dataset.batch(BATCH_SIZE)

    model.fit(
        train_dataset,
        steps_per_epoch=int(np.ceil(x/BATCH_SIZE)),
         epochs=EPOCHS,
         validation_data=val_dataset,
         validation_steps=10
            )
    test_dataset = tf.data.TextLineDataset("files/test_triplets.txt ")   
    test_dataset = test_dataset.map(
                    lambda line: get_test_dataset(line),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    num_test_samples = len(open("files/test_triplets.txt").readlines())
    inference_model = get_inference_model(model)
     
    test_dataset = test_dataset.batch(BATCH_SIZE)
    predictions = inference_model.predict(
        test_dataset,
        steps=int(np.ceil(num_test_samples/BATCH_SIZE)),
        verbose=1)
    np.savetxt("predictions.txt", predictions, fmt="%i")
    

def get_inference_model(model):
    dp, dn = distances(model.output)
    predictions = tf.cast(tf.greater_equal(dn, dp), tf.int8)
    return tf.keras.Model(inputs=model.inputs, outputs=predictions)

def load_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = img/255
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
   
    return img


def get_dataset(line):
    name = tf.strings.split(line)
    anc = load_image(tf.io.read_file('food/' + name[0] + '.jpg'))
    pos = load_image(tf.io.read_file('food/' + name[1] + '.jpg'))
    neg = load_image(tf.io.read_file('food/' + name[2] + '.jpg'))
    
    return tf.stack([anc, pos, neg], axis=0), 1
    
def get_test_dataset(line):
    name = tf.strings.split(line)
    anc = load_image(tf.io.read_file('food/' + name[0] + '.jpg'))
    pos = load_image(tf.io.read_file('food/' + name[1] + '.jpg'))
    neg = load_image(tf.io.read_file('food/' + name[2] + '.jpg'))
    
    return tf.stack([anc, pos, neg], axis=0)


def get_model():

    encoder = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    encoder.trainable = False

    embedding_model = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(DIM),
        tf.keras.layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=1))
    ])
    
    inputs = tf.keras.Input(shape=(3, IMG_SIZE, IMG_SIZE, 3))
    anc = inputs[:, 0, ...]
    pos = inputs[:, 1, ...]
    neg = inputs[:, 2, ...]
   
    outputs = tf.stack([embedding_model(encoder(anc)), embedding_model(encoder(pos)), embedding_model(encoder(neg))], axis=-1)
    outputs = tf.stack([embedding_model(encoder(anc)), embedding_model(encoder(pos)), embedding_model(encoder(neg))], axis=-1)
    net = tf.keras.Model(inputs=inputs, outputs=outputs)
    net.summary()
    return net

def triplet_loss(_, y_pred):
    anc, pos, neg = y_pred[...,0], y_pred[...,1], y_pred[...,2] 
    dp = tf.reduce_sum(tf.square(anc-pos), 1)
    dn = tf.reduce_sum(tf.square(anc-neg), 1)
    return tf.reduce_mean(tf.math.softplus(dp- dn))


def accuracy(_, embeddings):
    dp, dn = distances(embeddings)
    return tf.reduce_mean(tf.cast(tf.greater_equal(dn, dp), tf.float32))

def distances(embeddings):
    anc, pos, neg = embeddings[..., 0], embeddings[..., 1], embeddings[..., 2]
    dp = tf.reduce_sum(tf.square(anc - pos), 1)
    dn = tf.reduce_sum(tf.square(anc - neg), 1)
    return dp, dn

if __name__ == '__main__':
    main()