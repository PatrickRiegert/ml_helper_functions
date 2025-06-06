import tensorflow as tf

def train_data(train_data_name, lablel_mode):
  train_data_loaders = tf.keras.preprocessing.image_dataset_from_directory(directory=train_data_name, image_size=IMG_SIZE, label_mode=lablel_mode, batch_size=BATCH_SIZE)
  return train_data_loaders


def test_data(test_data_name, lablel_mode):
  test_data_loaders = tf.keras.preprocessing.image_dataset_from_directory(directory=test_data_name, image_size=IMG_SIZE, label_mode=lablel_mode, batch_size=BATCH_SIZE)
  return test_data_loaders


def data_augmentation_creater_efficient_net(rflip, rrot, rzoom, rheight, rwidth, data_aug_name, input):
  
  '''
  Only for EfficientNet
  '''
  data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip(rflip),
      tf.keras.layers.RandomRotation(rrot),
      tf.keras.layers.RandomZoom(rzoom),
      tf.keras.layers.RandomHeight(rheight),
      tf.keras.layers.RandomWidth(rwidth)],
      name=data_aug_name)
  augmented_tensor = data_augmentation(input)
  return augmented_tensor


def data_augmentation_creater(rflip, rrot, rzoom, rheight, rwidth, rsale, data_aug_name, input):

  '''
  Not for EfficientNet
  '''
  data_augmentation = tf.keras.Sequential([
      tf.keras.layers.RandomFlip(rflip),
      tf.keras.layers.RandomRotation(rrot),
      tf.keras.layers.RandomZoom(rzoom),
      tf.keras.layers.RandomHeight(rheight),
      tf.keras.layers.RandomWidth(rwidth),
      tf.keras.layers.Rescaling(rsale)],
      name=data_aug_name)
  augmented_tensor = data_augmentation(input)
  return augmented_tensor
