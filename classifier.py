import os
from sklearn.externals import joblib
import glob
from sklearn.utils import shuffle
from feature_extractor import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.model_selection import GridSearchCV
import sklearn.svm as svm


# Encapsulates classifier used by car detection
# If classifier is instantiated  and no "classifier.p" exits
# the classifier is created, trained and saved as "classifier.p" file
class Classifier:

    def __init__(self):

        self.picle_file = 'data/classifier.p'
        self.vehicle_file = 'data/vehicles/**/*.png'
        self.non_vehicle_file = 'data/non-vehicles/**/*.png'

        # load classifier from pickle file
        self.classifier, self.scaler = self.__load__()

    def is_not_valid(self):
        invalid = self.classifier is None or self.scaler is None
        return invalid

    def __load__(self):
        if os.path.isfile(self.picle_file):
            return joblib.load(self.picle_file)
        return None, None

    def __save__(self):
        joblib.dump((self.classifier, self.scaler), self.picle_file, compress=9)

    # Creates and trains a classifier on training data
    # should only called if classifier is not valid
    def create(self, color_space, feature_extractor):
        vehicles = []
        non_vehicles = []
        for v in glob.glob(self.vehicle_file, recursive=True):
            vehicles.append(v)
        for nv in glob.glob(self.non_vehicle_file, recursive=True):
            non_vehicles.append(nv)
        print("Vehicles: ", len(vehicles), ", Non vehicles: ", len(non_vehicles))

        vehicles = shuffle(vehicles)
        non_vehicles = shuffle(non_vehicles)
        min_len = min(len(vehicles), len(non_vehicles))
        vehicles = vehicles[:min_len]
        non_vehicles = non_vehicles[:min_len]

        # Define features & labels
        print("Extracting training data..")
        car_features = feature_extractor.extract_features_for_image_list(vehicles, color_space)
        no_car_features = feature_extractor.extract_features_for_image_list(non_vehicles, color_space)

        # Create an array stack of feature vectors
        X = np.vstack((car_features, no_car_features)).astype(np.float64)
        X_scaler = StandardScaler().fit(X)
        print(X.shape)
        X = X_scaler.transform(X)
        y = np.hstack((np.ones(len(car_features)), np.zeros(len(no_car_features))))

        # Split up data into randomized training and test sets
        rand_state = np.random.randint(0, 100)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rand_state)

        print("Training..")
        parameters = {'kernel': ['rbf']}
        svr = svm.SVC()
        clf = GridSearchCV(svr, parameters, n_jobs=4)
        t = time.time()
        clf.fit(X_train, y_train)
        print('Training time: ', round(time.time() - t, 2))
        print('Best params  : ', clf.best_params_)
        print('Best score  : ', clf.best_score_)
        print('Test accuracy: ', clf.score(X_test, y_test))
        print('Test predicts: ', clf.predict(X_test[0:15]))
        print('For labels   : ', y_test[0:15])

        # assign classifier and scaler
        self.classifier = clf
        self.scaler = X_scaler
        # save classifier to pickle file
        self.__save__()
        return

