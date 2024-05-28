import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import random
import os

def view_random_image(target_dir, target_class):
    target_loc = target_dir + target_class
    img_filename = random.choice(os.listdir(target_loc))
    img = mpimg.imread(target_loc + "/" + img_filename)
    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.title(target_class)
    plt.axis(False)