import os
import numpy as np
import tensorflow as tf

def load_images(dir_path, img_size=256):
    img_dims = [img_size, img_size]
    images = []
    for f_name in os.listdir(dir_path):
        img = tf.keras.utils.load_img(f"{dir_path}/{f_name}", target_size=img_dims)
        img = tf.keras.utils.img_to_array(img)
        images.append(img)
    return np.asarray(images)

def convert_to_npz(base_path, filename, new_size:int=256):
    """
    Takes in file_path
    returns the dataset for supervised learning tasks
    """
    trainA = load_images(base_path + "trainA/", 256)
    testA = load_images(base_path + "testA/", 256)
    trainB = load_images(base_path + "trainB/", 256)
    testB = load_images(base_path, "testB/", 256)
    
    domain_A_data = np.vstack([trainA, testA])
    domain_B_data = np.vstack([trainB, testB])

    np.savez_compressed(filename, domain_A_data, domain_B_data)
    print("Saved Dataset :", filename)

def load_domain_dataset(domain_loc):
    return tf.keras.utils.image_dataset_from_directory(
        domain_loc, 
        shuffle=True,
        batch_size=None,
        labels=None
    )