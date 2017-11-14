import feature_extractor as feat
from classifier import *
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

# ncapsulates heat map to filter false positives predicted by classifier.
class HeatMap:
    def __init__(self, image_size):
        self.history = []
        self.map = np.zeros((image_size[0], image_size[1]), np.float32)

    # Adds low-pass filter to heat
    def filter_heat(self, heatmap):
        # Append x values
        self.history.append(heatmap)

        # only 10 video frames will be stored
        if len(self.history) > 10:
            self.history.pop(0)

        mean_map = np.zeros(self.map.shape)
        mean_count = 0
        for map in self.history:
            if map is not None:
                mean_map += map
                mean_count += 1
        mean_map /= mean_count
        return mean_map

    def add_heat(self, boxes):
        for box in boxes:
            self.map[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1.0

    def apply_threshold(self, threshold):
        max_heat = np.max(self.map)
        if max_heat > 0.0:
            self.map /= max_heat
        self.map = self.filter_heat(self.map)
        self.map[self.map < threshold] = 0.0

    def get_labels(self):
        return label(self.map)


# Vehicle detector
class VehicleDetector:

    def __init__(self, classifier, color_space, hog_feature, img_size):
        self.clf = classifier
        self.color_space = color_space
        self.hog_feature = hog_feature
        self.heat_map = HeatMap(img_size)

        self.subwindows = [
           (400, 464, 1.0), # 64,64
           (416, 480, 1.0), # 64,64
           (400, 496, 1.5), # 96,96
           (432, 528, 1.5), # 96,96
           (400, 528, 2.0), # 128,128
           (432, 560, 2.0), # 128,128
           (400, 596, 3.0),  # 196,196
           (464, 660, 3.0)]  # 196,196

        self.box_colors = [
            (255, 0, 0),
            (255, 106, 0),
            (255, 216, 0),
            (182, 255, 0),
            (0, 255, 255),
            (72, 0, 255),
            (255, 0, 220)]

    # Draw label boxes based on given labels
    def draw_labeled_boxes(self, img, labels):
        rects = []
        # iterate through all detected cars
        for car_number in range(1, labels[1] + 1):
            # find pixels with each car_number label value
            nonzero = (labels[0] == car_number).nonzero()
            # Identify x and y values of those pixels
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])

            # define a bounding box based on min/max x and y
            bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
            rects.append(bbox)
            # draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (255,20,147), 6)
        return img, rects

    # Extract features for list of image regions
    def dectect_car_boxes(self, img):
        car_boxes = []
        for window in self.subwindows:
            boxes = self.find_car_boxes(img, window, 'ALL')
            car_boxes.append(boxes)
        # apparently this is the best way to flatten a list of lists
        car_boxes = [item for sublist in car_boxes for item in sublist]
        return car_boxes

    # Extract features in particular region of image (defined by window)
    # using hog sub-sampling and make predictions
    def find_car_boxes(self, img, window, hog_channel):
        # array of boxes where cars were detected
        boxes = []

        ystart = window[0]
        ystop = window[1]
        scale = window[2]
        pix_per_cell = self.hog_feature.pix_per_cell

        # clip image
        clip_img = img[ystart:ystop, :, :]

        # rescale clipped image id  required
        if scale != 1:
            clip_img = cv2.resize(clip_img, (np.int(clip_img.shape[1] / scale), np.int(clip_img.shape[0] / scale)))

        # select color space channel for HOG
        if hog_channel == 'ALL':
            ch1 = clip_img[:, :, 0]
            ch2 = clip_img[:, :, 1]
            ch3 = clip_img[:, :, 2]
        else:
            ch1 = clip_img[:, :, hog_channel]

        # Define blocks and steps as above
        nxblocks = (ch1.shape[1] // pix_per_cell) + 1  # -1
        nyblocks = (ch1.shape[0] // pix_per_cell) + 1  # -1

        # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
        window = 64
        nblocks_per_window = (window // pix_per_cell) - 1
        cells_per_step = 2  # Instead of overlap, define how many cells to step
        nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
        nysteps = (nyblocks - nblocks_per_window) // cells_per_step

        # Compute individual channel HOG features for the entire image (ony once)
        hog1 = self.hog_feature.get_features(ch1,  feature_vec=False)
        if hog_channel == 'ALL':
            hog2 = self.hog_feature.get_features(ch2, feature_vec=False)
            hog3 = self.hog_feature.get_features(ch3, feature_vec=False)

        for xb in range(nxsteps):
            for yb in range(nysteps):
                ypos = yb * cells_per_step
                xpos = xb * cells_per_step

                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                if hog_channel == 'ALL':
                    hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog_feat1

                xleft = xpos * pix_per_cell
                ytop = ypos * pix_per_cell

                # provide prediction
                test_features = self.clf.scaler.transform(np.hstack(hog_features).reshape(1, -1))
                test_prediction = self.clf.classifier.predict(test_features)

                if test_prediction == 1: # it's a car (it has label equal to 1)
                    xbox_left = np.int(xleft * scale)
                    ytop_draw = np.int(ytop * scale)
                    win_draw = np.int(window * scale)
                    boxes.append(
                        ((xbox_left, ytop_draw + ystart), (xbox_left + win_draw, ytop_draw + win_draw + ystart)))
        return boxes

    # Draws list of boxes (defined with left,top and right/bottom corners)
    @staticmethod
    def draw_boxes(img, bboxes, colors, thick=4):

        ncolors = len(colors)
        if ncolors == 0:
            return None
        # Make a copy of the image
        imcopy = np.copy(img)

        # Iterate through the bounding boxes
        i = 0
        for bbox in bboxes:
            color = colors[i]
            i = i + 1 if i + 1 < ncolors else 0
            cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
        return imcopy

    # Detect a car boxes and draws them in output image
    def run(self, img):

        # convert input image to setup color space (this one used by training of classifier too)
        match_img = feat.convert_image(img, self.color_space)

        # detect boxes representing recognized cars
        car_boxes = self.dectect_car_boxes(match_img)

        # add heat
        for i in range(len(car_boxes)):
            self.heat_map.add_heat(car_boxes)
        self.heat_map.apply_threshold(0.3)

        # get labels and draw boxes
        labels = self.heat_map.get_labels()
        draw_img, rect = self.draw_labeled_boxes(img, labels)

        """ Some images for writeup
        img_boxes =  VehicleDetector.draw_boxes(img, car_boxes, self.box_colors)
        f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(10, 7))
        f.subplots_adjust(hspace=.1, wspace=.1)
        ax1.set_title('Boxes')
        ax1.axis('off')
        ax1.imshow(img_boxes)
        ax2.set_title('Heat')
        ax2.axis('off')
        ax2.imshow(self.heat_map.map, cmap="hot")
        ax3.set_title('Label')
        ax3.axis('off')
        ax3.imshow(labels[0], cmap="gray")
        ax4.set_title('Detected box')
        ax4.axis('off')
        ax4.imshow(img)
        plt.show()
        """

        return draw_img


#-------------------------------------------
#           ***  Main logic  ***
#-------------------------------------------

if __name__ == '__main__':

    # setup color space and feature extractor
    color_space = 'YUV'
    hog_feature = feat.HOGFeature(orientation=11, pix_per_cell=16, cells_per_block=2)

    # load or if needed (only at first time) create and train classifier
    clf = Classifier()
    if clf.is_not_valid():
        clf.create(color_space, hog_feature)

    """
    # test pipline on test images process  test images
    for img_file in glob.glob('test_images/*.jpg'):
        img = feat.read_image_safe(img_file, color_space='RGB')
        veh_det = VehicleDetector(clf, color_space, hog_feature, img.shape)
        draw_img = veh_det.run(img)
        mpimg.imsave('output_images/{}'.format(os.path.basename(img_file)), draw_img, format='jpg')

    """

    # run implementation on test/project video
    video_file = 'test_video.mp4'
    #video_file = 'project_video.mp4'
    video_path = os.path.join('output_videos', video_file)

    clip = VideoFileClip(video_file)
    #clip = VideoFileClip(video_file).subclip(0,5)
    veh_detector = VehicleDetector(clf, color_space, hog_feature, (clip.h, clip.w, 3))
    new_clip = clip.fl_image(veh_detector.run)
    new_clip.write_videofile(video_path, audio=False)

    print("God help me !")






