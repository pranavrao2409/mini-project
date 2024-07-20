import os
import random
import cv2
from skimage import feature
from imutils import paths
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix

def quantify_image(image):
    features = feature.hog(image, orientations=9,
                           pixels_per_cell=(10, 10), cells_per_block=(2, 2),
                           transform_sqrt=True, block_norm="L1")
    return features

def load_split(path):
    imagePaths = list(paths.list_images(path))
    data = []
    labels = []
    for imagePath in imagePaths:
        label = imagePath.split(os.path.sep)[-2]
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        features = quantify_image(image)
        data.append(features)
        labels.append(label)
    return (np.array(data), np.array(labels))

def train_models(dataset):
    models = {
        "Rf": {
            "classifier": RandomForestClassifier(random_state=1),
            "accuracy": 0,
            "sensitivity": 0,
            "specificity": 0,
        },
        "Xgb": {
            "classifier": XGBClassifier(),
            "accuracy": 0,
            "sensitivity": 0,
            "specificity": 0,
        }
    }
    path = f"C:/vs_code/virtual enve/{dataset}"
    trainingPath = os.path.sep.join([path, "training"])
    testingPath = os.path.sep.join([path, "testing"])

    (trainX, trainY) = load_split(trainingPath)
    (testX, testY) = load_split(testingPath)

    le = LabelEncoder()
    trainY = le.fit_transform(trainY)
    testY = le.transform(testY)

    for model in models:
        models[model]["classifier"].fit(trainX, trainY)
        predictions = models[model]["classifier"].predict(testX)
        cm = confusion_matrix(testY, predictions).ravel()
        tn, fp, fn, tp = cm
        models[model]["accuracy"] = (tp + tn) / float(cm.sum())
        models[model]["sensitivity"] = tp / float(tp + fn)
        models[model]["specificity"] = tn / float(tn + fp)
    return models

def test_prediction(model, testingPath):
    testingPaths = list(paths.list_images(testingPath))
    output_images = []
    for _ in range(15):
        image = cv2.imread(random.choice(testingPaths))
        output = image.copy()
        output = cv2.resize(output, (128, 128))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (200, 200))
        image = cv2.threshold(image, 0, 255,
                              cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

        features = quantify_image(image)
        preds = model.predict([features])
        label = "Parkinsons" if preds[0] else "Healthy"
        color = (0, 255, 0) if label == "Healthy" else (0, 0, 255)
        cv2.putText(output, label, (3, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    color, 2)
        output_images.append(output)
    return output_images

if __name__ == "__main__":
    spiralModels = train_models('spiral')
    waveModels = train_models('wave')

    print("Random Forrest vs XGBoost Classifier\n\n")
    for metric in ("accuracy", "sensitivity", "specificity"):
        print(f"{metric.capitalize()}: ")
        print("Random Forrest={:.2f}%, XGBoost={:.2f}% \n".format(
            spiralModels['Rf'][metric]*100, spiralModels['Xgb'][metric]*100))
        print("Random Forrest vs XGBoost Classifier\n\n")

    for metric in ("accuracy", "sensitivity", "specificity"):
        print(f"{metric.capitalize()}: ")
        print("Random Forrest={:.2f}%, XGBoost={:.2f}% \n".format(
            waveModels['Rf'][metric]*100, waveModels['Xgb'][metric]*100))

    testingPath = os.path.sep.join(["C:/vs_code/virtual enve/spiral", "testing"])
    output_images = test_prediction(spiralModels['Rf']['classifier'], testingPath)
    plt.figure(figsize=(20, 20))
    for i in range(len(output_images)):
        plt.subplot(5, 5, i+1)
        plt.imshow(output_images[i])
        plt.axis("off")
    plt.show()

    testingPath = os.path.sep.join(["C:/vs_code/virtual enve/wave", "testing"])
    output_images = test_prediction(waveModels['Rf']['classifier'], testingPath)
    plt.figure(figsize=(20, 20))
    for i in range(len(output_images)):
        plt.subplot(5, 5, i+1)
        plt.imshow(output_images[i])
        plt.axis("off")
    plt.show()
