
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

IMG_SIZE= 96
BATCH_SIZE=12
EPOCHS=5


def main():
    
    
    num_train_samples = len(open("files/train_triplets.txt").readlines())
    train_dataset = tf.data.TextLineDataset("files/train_triplets.txt")
    train_dataset = train_dataset.map(
                       lambda line: get_train_dataset(line),
                       num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    
    model = get_model()
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss=triplet_loss,
                  metrics=[accuracy])
    train_dataset = train_dataset.shuffle(1024).batch(BATCH_SIZE).repeat()

    history = model.fit(
        train_dataset,
        steps_per_epoch=int(num_train_samples/BATCH_SIZE),
         epochs=EPOCHS,
            )
    test_dataset = tf.data.TextLineDataset("files/test_triplets.txt")
    test_dataset = test_dataset.map(
                    lambda line: get_test_dataset(line),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
    
    num_test_samples = len(open("files/test_triplets.txt").readlines())
    predictions = model.predict(
        test_dataset,
        steps=int(num_test_samples/BATCH_SIZE),
        verbose=1)
    np.savetxt('predictions.txt', predictions, fmt='%i')
    



def load_image(img):
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
   
    return img


def get_train_dataset(line):
    name = tf.strings.split(line)
    anc = load_image(tf.io.read_file('food/' + name[0] + '.jpg'))
    pos = load_image(tf.io.read_file('food/' + name[1] + '.jpg'))
    neg = load_image(tf.io.read_file('food/' + name[2] + '.jpg'))
    
    return tf.stack([anc, pos, neg], axis=0), 1
    
def get_test_dataet(line):
    name = tf.strings.split(line)
    anc = load_image(tf.io.read_file('food/' + name[0] + '.jpg'))
    pos = load_image(tf.io.read_file('food/' + name[1] + '.jpg'))
    neg = load_image(tf.io.read_file('food/' + name[2] + '.jpg'))
    
    return tf.stack([anc, pos, neg], axis=0)


def get_model(freeze=True):
    # mobilenet_weights_path = 'resnet50v2_weights_tf_dim_ordering_tf_kernels_notop.h5'
    inputs = tf.keras.Input(shape=(3, IMG_SIZE, IMG_SIZE, 3))
    encoder = tf.keras.applications.MobileNetV2(
        include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    encoder.trainable = not freeze
    decoder = tf.keras.Sequential([
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dense(128),
        tf.keras.layers.Lambda(
            lambda t: tf.math.l2_normalize(t, axis=1))
    ])
    anc, pos, neg = inputs[:, 0, ...], inputs[:, 1, ...], inputs[:, 2, ...]
   
    embeddings = tf.stack([decoder(encoder(anc)), decoder(encoder(pos)),decoder(encoder(neg))], axis=-1)
    tripl= tf.keras.Model(inputs=inputs, outputs=embeddings)
    triple_siamese.summary()
    return triple_siamese


def compute_distances_from_embeddings(embeddings):
    anchor, truthy, falsy = embeddings[..., 0], embeddings[..., 1], embeddings[..., 2]
    distance_truthy = tf.reduce_sum(tf.square(anchor - truthy), 1)
    distance_falsy = tf.reduce_sum(tf.square(anchor - falsy), 1)
    return distance_truthy, distance_falsy



def triplet_loss(_, embeddings):
    dis_pos, dis_neg = loss(embeddings)
    return tf.reduce_mean(tf.math.softplus(dis_pos-dis_neg))


def loss(embeddings):
    anc, pos, neg = embeddings[...,0], embeddings[...,1], embeddings
    dis_pos = tf.reduce_sum(tf.square(anc-pos), 1)
    dis_neg = tf.reduce_sum(tf.square(anc-neg), 1)
    return dis_pos, dis_neg


def accuracy(_, embeddings):
    dis_pos, dis_neg = compute_distances_from_embeddings(embeddings)
    return tf.reduce_mean(tf.cast(tf.greater_equal(dis_pos, dis_pos), tf.float32))



if __name__ == '__main__':
    main()