from model.model import ImageClassifier
from sklearn.metrics import classification_report


#num of classes in our dataset
num_classes = 10

#instaniate the model class
cls = ImageClassifier()

#load the trained saved model
model = cls.load_model()

#predicted classes
predicted_classes = model.predict_classes(cls.x_test)

# show Classification report
target_names = ["Class {}".format(i) for i in range(num_classes)]
print(classification_report(cls.y_test, predicted_classes, target_names=target_names))