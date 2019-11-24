import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv2D, concatenate, Flatten, Conv2DTranspose, Reshape, Dense
from tensorflow.keras import Model
import numpy as np
from matplotlib import pyplot as plt
import tqdm

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

def normalize(img):  # -1 ~ 1
    return (img - 127.5) / 127.5

def denormalize(img):
    return (img * 127.5 + 127.5)

# do data process in dataset map function
x_train = normalize(x_train[..., tf.newaxis])
x_test = normalize(x_test[..., tf.newaxis])

batch_size = 32
epochs = 10

y_train = tf.one_hot(y_train, depth=10)
y_test = tf.one_hot(y_test, depth=10)

x_train = (x_train, y_train)
x_test = (x_test, y_test)
features, labels = (np.random.sample((100, 2)), np.random.sample((100, 1)))


# TODO: optimize this dataset (map function)
train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(batch_size)
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(10000).batch(batch_size)


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv_1 = Conv2D(64, kernel_size=(4, 4), strides=2, padding='same', activation='relu')
        self.conv_2 = Conv2D(128, kernel_size=(4, 4), strides=2, padding='same', activation='relu')
        self.flatten = Flatten()
        self.dense_1 = Dense(1, activation='sigmoid')

    def call(self, img, condition):
        img = self.conv_1(img)
        img = self.conv_2(img)
        feature_vector = self.flatten(img)
        concated_layer = concatenate([feature_vector, condition])
        logit = self.dense_1(concated_layer)
        return logit


discriminator = Discriminator()


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense_1 = Dense(7 * 7 * 128, activation='relu')
        self.reshape = Reshape((7, 7, 128))
        self.deconv_1 = Conv2DTranspose(128, kernel_size=(4, 4), strides=2, padding='same', activation='relu')
        self.deconv_2 = Conv2DTranspose(64, kernel_size=(4, 4), strides=2, padding='same', activation='relu')
        self.deconv_3 = Conv2DTranspose(1, kernel_size=(1, 1), strides=1, padding='valid', activation='tanh')

    def call(self, latent, condition):
        conditioned_latent = concatenate([latent, condition])
        x = self.dense_1(conditioned_latent)
        x = self.reshape(x)
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        return x


generator = Generator()
cross_entrophy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
latent = tf.random.normal([32, 100])
condition = y_train[:32]

fake_img = generator(latent, condition)
fake_img[0].shape
plt.imshow(fake_img[0, :, :, 0], cmap='gray')


def get_disc_loss(fake_output, real_output):
    real = cross_entrophy(tf.ones_like(real_output), real_output)
    fake = cross_entrophy(tf.zeros_like(fake_output), fake_output)
    return real + fake


def get_gen_loss(fake_output):
    return cross_entrophy(tf.ones_like(fake_output), fake_output)

# @tf.function
def train_step(data):
    with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
        real_img, label = data
        latent = tf.random.normal([32, 100])
        fake_img = generator(latent, label)

        real_output = discriminator(real_img, label)
        fake_output = discriminator(fake_img, label)
        d_loss = get_disc_loss(fake_output, real_output)
        g_loss = get_gen_loss(fake_output)

        g_gradients = g_tape.gradient(g_loss, generator.trainable_variables)
        d_gradients = d_tape.gradient(d_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))

        # print(f"g_loss : {g_loss} d_loss : {d_loss}")

def make_images(generator, labels,latent,epoch):
    img = generator(latent, labels)

    fig = plt.figure(figsize=(10, 10))
    for i in range(10):
        plt.subplot(1, 10,i+1)
        plt.imshow(img[i,:,:,0],cmap='gray')
        plt.savefig(f"-{epoch}.png")
    plt.show()

for epoch in range(epochs):
    step_num = 0
    for data in tqdm(train_dataset, desc=f"{epoch}"):
        train_step(data)
        if epoch%10 == 0:
            print(f"epoch : {epoch}")
            latent = tf.random.normal([10, 100])
            labels = np.zeros((10, 10))
            np.fill_diagonal(labels,1)
            make_images(generator,labels,latent,epoch)

