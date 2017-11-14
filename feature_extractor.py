import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import cv2
from skimage.feature import hog


# Converts an image into particular color space
def convert_image(image, color_space='RGB'):
    if color_space != 'RGB':
        if color_space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            return cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            return cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    return image

# Reads and image and scales it from 0 to 255 if required
# to avoid different image scaling dpending on image format (png or jpeg)
def read_image_safe(file, color_space='RGB'):
    image = mpimg.imread(file)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    return convert_image(image, color_space)



# Encapsulates extraction of Histogram of Oriented Gradients (HOG)
# feature from an image
class HOGFeature:

    def __init__(self, orientation=11, pix_per_cell=16, cells_per_block=2):
        self.orientation = orientation
        self.pix_per_cell = pix_per_cell
        self.cells_per_block = cells_per_block

    # Extract features for a given image.
    def get_features(self, image, feature_vec=False):
        features = hog(image,
                       orientations=self.orientation,
                       pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                       cells_per_block=(self.cells_per_block, self.cells_per_block),
                       transform_sqrt=False,
                       visualise=False,
                       feature_vector=feature_vec,
                       block_norm='L2')
        return features

    # Extract features to train classifier, use feature_vec=True
    # Return the data as a feature vector by calling .ravel()
    # on the result just before returning.
    def extract_features(self, image, hog_channel='ALL'):

        if hog_channel == 'ALL':
            hog_features = []
            for ch in range(image.shape[2]):
                # extract features for particular channel and add them to hog features
                features = self.get_features(image[:, :, ch], feature_vec=True)
                hog_features.append(features)

            hog_features = np.ravel(hog_features)
        else:
            hog_features = self.get_features(image[:, :, hog_channel], feature_vec=True)

        return hog_features

    # Extract features to train classifier for list of image files
    def extract_features_for_image_list(self, image_files, color_space, hog_channel='ALL'):
        features = []
        for image_file in image_files:
            image = read_image_safe(image_file, color_space)
            img_features = self.extract_features(image, hog_channel)
            features.append(img_features)
        return features

    # Extract features for a given image and provide HOG image
    def get_features_and_hog_image(self, image):
        # hog_channel is ignored
        features, hog_image = hog(image,
                                  orientations = self.orientation,
                                  pixels_per_cell=(self.pix_per_cell, self.pix_per_cell),
                                  cells_per_block=(self.cells_per_block, self.cells_per_block),
                                  transform_sqrt=False,
                                  visualise=True,
                                  feature_vector=True,
                                  block_norm='L2')
        return features, hog_image


#---------------------------------------------
#             Testing
#---------------------------------------------

# For testing purpose
if __name__ == '__main__':

    car_file = "data/vehicles/GTI_MiddleClose/image0068.png"
    non_car_file = "data/non-vehicles/extra1002.png"

    """
    #img = read_image_safe(car_file, 'YUV')
    img = read_image_safe(non_car_file, 'YUV')

    f, ax = plt.subplots(2, 3, figsize=(7, 7), frameon=False)
    f.subplots_adjust(hspace=0.00, wspace=0.1)
    feat_ex = HOGFeature(orientation=11, pix_per_cell=16, cells_per_block=2)

    v, vis_img = feat_ex.get_features_and_hog_image(img[:, :, 0])
    ax[0, 0].set_title('Y channel')
    ax[0, 0].axis('off')
    ax[0, 0].imshow(img[:, :, 0], cmap="gray")
    plt.title('HOG Features')
    ax[1, 0].set_title('Y HOG')
    ax[1, 0].axis('off')
    ax[1, 0].imshow(vis_img, cmap="gray")

    _, vis_img = feat_ex.get_features_and_hog_image(img[:, :, 1])
    ax[0, 1].set_title('U channel')
    ax[0, 1].axis('off')
    ax[0, 1].imshow(img[:, :, 1], cmap="gray")
    ax[1, 1].set_title('U HOG')
    ax[1, 1].axis('off')
    ax[1, 1].imshow(vis_img, cmap="gray")

    _, vis_img = feat_ex.get_features_and_hog_image(img[:, :, 2])
    ax[0, 2].set_title('V channel')
    ax[0, 2].axis('off')
    ax[0, 2].imshow(img[:, :, 2], cmap="gray")
    ax[1, 2].set_title('V HOG')
    ax[1, 2].axis('off')
    ax[1, 2].imshow(vis_img, cmap="gray")

    plt.show()
    """

    img1 = read_image_safe(car_file, 'RGB')
    img2 = read_image_safe(non_car_file, 'RGB')

    # Visualize
    f, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(7, 7))
    f.subplots_adjust(hspace=.4, wspace=.2)
    ax1.imshow(img1)
    ax1.axis('off')
    ax1.set_title('Car image', fontsize=16)
    ax2.imshow(img2)
    ax2.set_title('Non car image', fontsize=16)
    ax2.axis('off')
    plt.show()
    print('...')


    print("the end")


