import numpy as np


from sklearn.datasets import load_digits
from sklearn.pipeline import Pipeline

# http://scikit-learn.org/stable/whats_new.html#version-0-18
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.externals import joblib


digits = load_digits(10)

size = 1500
train_X = digits.data[:size]
train_y = digits.target[:size]

test_X = digits.data[size:]
test_y = digits.target[size:]

# pipeline
pipeline = Pipeline([
    ('standard_scaler', StandardScaler()),
    ('svm', SVC())])

search_size = 5
params = {
        'svm__C': np.logspace(0, 2, search_size),
        'svm__gamma': np.logspace(-3, 0, search_size),
}

# grid search
clf = GridSearchCV(pipeline, params)
clf.fit(train_X, train_y)

pred = clf.predict(test_X)


# reporting result
print(classification_report(test_y, pred))
print(confusion_matrix(test_y, pred))

# save model
joblib.dump(clf, 'clf.pkl')
