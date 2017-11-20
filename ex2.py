from scipy import misc
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


"""
Description: Implementing the Locally Sensitive Algorithm (LSH) from lecture #10 slides
"""

# Loads image and shows it
def load_image(filename):
    image = mpimg.imread(filename)
    plt.imshow(image)
    return image
    
# 1. Converts image to grayscale
def rgb2gray(image):
    return np.mean(image[...,:3], -1)

# 2. Resizes image
def resize(image, x, y):
    new_size = misc.imresize(image, (x,y))
    plt.imshow(new_size)
    return new_size

# 3. Compares adjacent pixel values
def pixel_compare(image):
    x,y = image.shape
    new_array = np.zeros((x,y), dtype=np.int)
    for i in range(0,x):
        for j in range(0,y-1):
            if (image[i,j] > image[i,j+1]):
                new_array[i,j] = 1
    return new_array

# 4. Create Hash
def hash_it2(differences):
    print("\nHash ID is: ")
    for difference in differences:
        decimal_value = 0
        hex_string = []
        for index, value in enumerate(difference):
            if value:
                decimal_value += 2**(index % 8)
            if (index % 8) == 7:
               hex_string.append(hex(decimal_value)[2:].rjust(2, '0'))
               decimal_value = 0
        print( ''.join(hex_string),)

# Loading images    
img1 = load_image('pup.png')            # same
img2 = load_image('pup_redball.png')    # almost the same
img3 = load_image('pup_diff.png')       # very different
img4 = load_image('bug.png')            # same
img5 = load_image('bug_dot.png')        # almost the same
img6 = load_image('bug_antenna.png')    # very different

image_list = [img1, img2, img3, img4, img5, img6]

# Processing and Printing Results
for i in range(len(image_list)):
    hash_it2(pixel_compare(resize(rgb2gray(image_list[i]), 9, 8)))




