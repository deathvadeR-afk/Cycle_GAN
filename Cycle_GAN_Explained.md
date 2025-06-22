# Cycle GAN Implementation Explained

This document provides explanations for each code cell in the Cycle GAN implementation for horse-to-zebra image translation.

## Installation and Imports

```python
!pip install git+https://github.com/tensorflow/examples.git
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow_examples.models.pix2pix import pix2pix
import os
import time
import matplotlib.pyplot as plt
from IPython.display import clear_output
```

This cell installs the TensorFlow examples repository which contains the pix2pix implementation that will be used for the Cycle GAN. It then imports necessary libraries:
- TensorFlow and TensorFlow Datasets for deep learning and dataset handling
- pix2pix module which contains pre-built GAN architectures
- OS and time for file operations and timing
- Matplotlib for visualization
- IPython's clear_output for clearing notebook output during training

## Loading the Dataset

```python
AUTOTUNE = tf.data.AUTOTUNE
dataset, metadata = tfds.load('cycle_gan/horse2zebra',
                              with_info=True, as_supervised=True)

train_horses, train_zebras = dataset['trainA'], dataset['trainB']
test_horses, test_zebras = dataset['testA'], dataset['testB']
```

This cell:
- Sets AUTOTUNE to let TensorFlow automatically determine the best number of parallel calls for data loading
- Loads the horse2zebra dataset from TensorFlow Datasets
- Separates the dataset into training and testing sets for both horses and zebras
- 'trainA' contains horse images, 'trainB' contains zebra images

## Setting Constants

```python
BUFFER_SIZE = 1000
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
```

This cell defines important constants:
- BUFFER_SIZE: Size of the buffer used when shuffling the dataset
- BATCH_SIZE: Number of images processed in each training step (1 for this implementation)
- IMG_WIDTH and IMG_HEIGHT: Dimensions to which all images will be resized (256x256 pixels)

## Image Processing Functions

```python
def random_crop(image):
  cropped_image = tf.image.random_crop(
      image, size=[IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image
```

This function randomly crops an image to the specified height and width (256x256) while preserving all 3 color channels.

```python
# normalizing the images to [-1, 1]
def normalize(image):
  image = tf.cast(image, tf.float32)
  image = (image / 127.5) - 1
  return image
```

This function normalizes pixel values from the standard [0, 255] range to [-1, 1], which is a common preprocessing step for GANs:
1. Converts the image to float32 data type
2. Divides by 127.5 and subtracts 1 to scale from [0, 255] to [-1, 1]

```python
def random_jitter(image):
  # resizing to 286 x 286 x 3
  image = tf.image.resize(image, [286, 286],
                          method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  # randomly cropping to 256 x 256 x 3
  image = random_crop(image)

  # random mirroring
  image = tf.image.random_flip_left_right(image)

  return image
```

This function applies data augmentation through "jittering":
1. Resizes the image to 286x286 using nearest neighbor interpolation
2. Randomly crops it back to 256x256, effectively creating a random position/zoom effect
3. Randomly flips the image horizontally (left to right)
These augmentations help the model generalize better by creating variations of the training data.

```python
def preprocess_image_train(image, label):
  image = random_jitter(image)
  image = normalize(image)
  return image
```

This function applies the full preprocessing pipeline for training images:
1. Applies random jitter (resize, crop, flip) for data augmentation
2. Normalizes the pixel values to [-1, 1]

```python
def preprocess_image_test(image, label):
  image = normalize(image)
  return image
```

This function preprocesses test images by only normalizing them without applying jitter, as test images should remain consistent.

## Preparing the Datasets

```python
train_horses = train_horses.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

train_zebras = train_zebras.cache().map(
    preprocess_image_train, num_parallel_calls=AUTOTUNE).shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_horses = test_horses.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)

test_zebras = test_zebras.map(
    preprocess_image_test, num_parallel_calls=AUTOTUNE).cache().shuffle(
    BUFFER_SIZE).batch(BATCH_SIZE)
```

This cell prepares the datasets for training and testing:
1. For training datasets (horses and zebras):
   - Caches the dataset to improve performance
   - Maps the preprocess_image_train function to each image
   - Uses parallel calls with AUTOTUNE for efficient processing
   - Shuffles the data with the specified buffer size
   - Batches the data according to the batch size

2. For testing datasets:
   - Maps the preprocess_image_test function (no jitter)
   - Caches, shuffles, and batches similarly to the training data

## Visualizing Sample Images

```python
sample_horse = next(iter(train_horses))
sample_zebra = next(iter(train_zebras))
```

This cell extracts a single sample from each training dataset to use for visualization.

```python
plt.subplot(121)
plt.title('Horse')
plt.imshow(sample_horse[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Horse with random jitter')
plt.imshow(random_jitter(sample_horse[0]) * 0.5 + 0.5)
```

This cell creates a side-by-side visualization:
1. Left: Original horse image (rescaled from [-1, 1] to [0, 1] for display)
2. Right: The same horse image with random jitter applied
This helps visualize the effect of the data augmentation.

```python
plt.subplot(121)
plt.title('Zebra')
plt.imshow(sample_zebra[0] * 0.5 + 0.5)

plt.subplot(122)
plt.title('Zebra with random jitter')
plt.imshow(random_jitter(sample_zebra[0]) * 0.5 + 0.5)
```

Similar to the previous cell, this creates a side-by-side visualization of a zebra image before and after applying random jitter.

## Creating the Generators and Discriminators

```python
OUTPUT_CHANNELS = 3

generator_g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
generator_f = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')

discriminator_x = pix2pix.discriminator(norm_type='instancenorm', target=False)
discriminator_y = pix2pix.discriminator(norm_type='instancenorm', target=False)
```

This cell creates the four neural networks needed for the Cycle GAN:
1. generator_g: Transforms horses to zebras (X→Y)
2. generator_f: Transforms zebras to horses (Y→X)
3. discriminator_x: Determines if an image is a real horse or a generated horse
4. discriminator_y: Determines if an image is a real zebra or a generated zebra

All models use instance normalization, which is preferred for style transfer tasks. The generators use a U-Net architecture which is effective for image-to-image translation.

## Testing the Untrained Generators

```python
to_zebra = generator_g(sample_horse)
to_horse = generator_f(sample_zebra)
plt.figure(figsize=(8, 8))
contrast = 8

imgs = [sample_horse, to_zebra, sample_zebra, to_horse]
title = ['Horse', 'To Zebra', 'Zebra', 'To Horse']

for i in range(len(imgs)):
  plt.subplot(2, 2, i+1)
  plt.title(title[i])
  if i % 2 == 0:
    plt.imshow(imgs[i][0] * 0.5 + 0.5)
  else:
    plt.imshow(imgs[i][0] * 0.5 * contrast + 0.5)
plt.show()
```

This cell tests the untrained generators and visualizes the results:
1. Passes a horse image through generator_g to create a "fake" zebra
2. Passes a zebra image through generator_f to create a "fake" horse
3. Displays a 2x2 grid of images: original horse, generated zebra, original zebra, generated horse
4. Applies contrast enhancement to the generated images to make patterns more visible

Since the generators are untrained at this point, the generated images won't look realistic.

## Testing the Untrained Discriminators

```python
plt.figure(figsize=(8, 8))

plt.subplot(121)
plt.title('Is a real zebra?')
plt.imshow(discriminator_y(sample_zebra)[0, ..., -1], cmap='RdBu_r')

plt.subplot(122)
plt.title('Is a real horse?')
plt.imshow(discriminator_x(sample_horse)[0, ..., -1], cmap='RdBu_r')

plt.show()
```

This cell visualizes the output of the untrained discriminators:
1. Shows how discriminator_y evaluates a real zebra image
2. Shows how discriminator_x evaluates a real horse image
The output is displayed as a heatmap where red/blue regions indicate the discriminator's assessment of whether different parts of the image look real or fake.

## Setting Up Loss Functions

```python
LAMBDA = 10
```

This cell defines LAMBDA, a hyperparameter that controls the weight of the cycle consistency loss.

```python
loss_obj = tf.keras.losses.BinaryCrossentropy(from_logits=True)
```

This creates the binary cross-entropy loss function that will be used for the adversarial loss. The `from_logits=True` parameter indicates that the discriminator outputs raw logits rather than probabilities.

```python
def discriminator_loss(real, generated):
  real_loss = loss_obj(tf.ones_like(real), real)

  generated_loss = loss_obj(tf.zeros_like(generated), generated)

  total_disc_loss = real_loss + generated_loss

  return total_disc_loss * 0.5
```

This function calculates the discriminator loss:
1. real_loss: Loss for classifying real images as real (target = 1)
2. generated_loss: Loss for classifying generated images as fake (target = 0)
3. The total loss is the sum of these two components, multiplied by 0.5 to slow down the discriminator training relative to the generator

```python
def generator_loss(generated):
  return loss_obj(tf.ones_like(generated), generated)
```

This function calculates the generator's adversarial loss:
- It measures how well the generator fooled the discriminator
- The target is 1 (trying to make the discriminator classify generated images as real)

```python
def calc_cycle_loss(real_image, cycled_image):
  loss1 = tf.reduce_mean(tf.abs(real_image - cycled_image))

  return LAMBDA * loss1
```

This function calculates the cycle consistency loss:
1. Measures the absolute difference between the original image and the cycled image
2. Multiplies by LAMBDA to weight this loss component
3. This loss ensures that if we translate an image from domain X to Y and back to X, we should get the original image

```python
def identity_loss(real_image, same_image):
  loss = tf.reduce_mean(tf.abs(real_image - same_image))
  return LAMBDA * 0.5 * loss
```

This function calculates the identity loss:
1. Measures how well a generator preserves images that are already in its output domain
2. For example, if we pass a zebra image to the horse-to-zebra generator, it should return the same zebra
3. This helps preserve color and composition between the input and output
4. It's weighted by LAMBDA * 0.5

## Setting Up Optimizers

```python
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
```

This cell creates Adam optimizers for all four networks:
1. Learning rate of 2e-4 (0.0002)
2. beta_1 of 0.5 (instead of the default 0.9)
These settings are commonly used for GANs as they help with training stability.

## Setting Up Checkpoints

```python
checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(generator_g=generator_g,
                           generator_f=generator_f,
                           discriminator_x=discriminator_x,
                           discriminator_y=discriminator_y,
                           generator_g_optimizer=generator_g_optimizer,
                           generator_f_optimizer=generator_f_optimizer,
                           discriminator_x_optimizer=discriminator_x_optimizer,
                           discriminator_y_optimizer=discriminator_y_optimizer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint)
  print ('Latest checkpoint restored!!')
```

This cell sets up checkpointing to save and restore model progress:
1. Creates a checkpoint object that includes all models and optimizers
2. Creates a checkpoint manager that will save up to 5 most recent checkpoints
3. Attempts to restore the latest checkpoint if one exists
This allows training to be resumed from where it left off if interrupted.

## Setting Training Parameters

```python
EPOCHS = 200
```

This cell sets the number of training epochs to 200.

## Creating Image Generation Function

```python
def generate_images(model, test_input):
  prediction = model(test_input)

  plt.figure(figsize=(12, 12))

  display_list = [test_input[0], prediction[0]]
  title = ['Input Image', 'Predicted Image']

  for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()
```

This function generates and displays translated images:
1. Takes a model (generator) and a test input
2. Generates a prediction using the model
3. Displays the input and predicted images side by side
4. Rescales the images from [-1, 1] to [0, 1] for proper display

## Defining the Training Step

```python
@tf.function
def train_step(real_x, real_y):
  # persistent is set to True because the tape is used more than
  # once to calculate the gradients.
  with tf.GradientTape(persistent=True) as tape:
    # Generator G translates X -> Y
    # Generator F translates Y -> X.

    fake_y = generator_g(real_x, training=True)
    cycled_x = generator_f(fake_y, training=True)

    fake_x = generator_f(real_y, training=True)
    cycled_y = generator_g(fake_x, training=True)

    # same_x and same_y are used for identity loss.
    same_x = generator_f(real_x, training=True)
    same_y = generator_g(real_y, training=True)

    disc_real_x = discriminator_x(real_x, training=True)
    disc_real_y = discriminator_y(real_y, training=True)

    disc_fake_x = discriminator_x(fake_x, training=True)
    disc_fake_y = discriminator_y(fake_y, training=True)

    # calculate the loss
    gen_g_loss = generator_loss(disc_fake_y)
    gen_f_loss = generator_loss(disc_fake_x)

    total_cycle_loss = calc_cycle_loss(real_x, cycled_x) + calc_cycle_loss(real_y, cycled_y)

    # Total generator loss = adversarial loss + cycle loss
    total_gen_g_loss = gen_g_loss + total_cycle_loss + identity_loss(real_y, same_y)
    total_gen_f_loss = gen_f_loss + total_cycle_loss + identity_loss(real_x, same_x)

    disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
    disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

  # Calculate the gradients for generator and discriminator
  generator_g_gradients = tape.gradient(total_gen_g_loss,
                                        generator_g.trainable_variables)
  generator_f_gradients = tape.gradient(total_gen_f_loss,
                                        generator_f.trainable_variables)

  discriminator_x_gradients = tape.gradient(disc_x_loss,
                                            discriminator_x.trainable_variables)
  discriminator_y_gradients = tape.gradient(disc_y_loss,
                                            discriminator_y.trainable_variables)

  # Apply the gradients to the optimizer
  generator_g_optimizer.apply_gradients(zip(generator_g_gradients,
                                            generator_g.trainable_variables))

  generator_f_optimizer.apply_gradients(zip(generator_f_gradients,
                                            generator_f.trainable_variables))

  discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients,
                                                discriminator_x.trainable_variables))

  discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients,
                                                discriminator_y.trainable_variables))
```

This function defines a single training step, decorated with @tf.function for faster execution:

1. Forward passes:
   - Generator G converts horses (X) to zebras (Y)
   - Generator F converts zebras (Y) to horses (X)
   - Cycle consistency: X → G(X) → F(G(X)) should equal X
   - Identity mapping: F(X) should equal X, G(Y) should equal Y
   - Discriminators evaluate real and fake images

2. Loss calculation:
   - Generator adversarial losses: How well generators fool discriminators
   - Cycle consistency loss: How well the original image is reconstructed after a full cycle
   - Identity loss: How well generators preserve images already in their target domain
   - Discriminator losses: How well discriminators distinguish real from fake

3. Gradient calculation and optimization:
   - Calculates gradients for all four networks
   - Applies gradients using the respective optimizers

The persistent=True parameter in GradientTape allows multiple gradient calculations from the same tape.

## Training Loop

```python
for epoch in range(EPOCHS):
  start = time.time()

  n = 0
  for image_x, image_y in tf.data.Dataset.zip((train_horses, train_zebras)):
    train_step(image_x, image_y)
    if n % 10 == 0:
      print ('.', end='')
    n += 1

  clear_output(wait=True)
  # Using a consistent image (sample_horse) so that the progress of the model
  # is clearly visible.
  generate_images(generator_g, sample_horse)

  if (epoch + 1) % 5 == 0:
    ckpt_save_path = ckpt_manager.save()
    print ('Saving checkpoint for epoch {} at {}'.format(epoch+1,
                                                         ckpt_save_path))

  print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                      time.time()-start))
```

This cell contains the main training loop:

1. For each epoch:
   - Tracks the start time
   - Zips the horse and zebra datasets together to get paired samples
   - Calls train_step for each pair of images
   - Prints a dot every 10 steps to show progress
   - Clears the output and generates a sample image to visualize progress
   - Saves a checkpoint every 5 epochs
   - Prints the time taken for the epoch

The same sample horse image is used for visualization throughout training to clearly show the model's progress.

## Testing the Trained Model

```python
# Run the trained model on the test dataset
for inp in test_horses.take(5):
  generate_images(generator_g, inp)
```

This final cell tests the trained horse-to-zebra generator on 5 images from the test dataset:
1. Takes 5 test horse images
2. Passes each through the generator_g model
3. Displays the original horse and generated zebra side by side for each

This demonstrates how well the model generalizes to unseen images.
