# dataset

from sklearn.datasets import load_digits

digits = load_digits(10)
# print(digits.data)
# print(digits.target)
# print(digits.data.shape)

size = 1500
train_X = digits.data[:size]
train_Y = digits.target[:size]

test_X = digits.data[size:]
test_Y = digits.target[size:]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()

# fit
lr.fit(train_X, train_Y)

# pred
pred = lr.predict(test_X)
# print(pred)

from sklearn.metrics import confusion_matrix

res = confusion_matrix(test_Y, pred, labels=digits.target_names)

print(res)
