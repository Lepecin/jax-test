from models import ClassModel
from datasets import dataset_train, dataset_test


X = dataset_train().drop(columns=["120_aa", "121_aa"]).to_numpy()
y = dataset_train()[["121_aa"]].to_numpy()

model = ClassModel(X.shape[1], y.max(), True, regularisation=0.1, epochs=500)


model.loss(X, y)

history = model.fit(X, y)

X_test = dataset_test().drop(columns=["120_aa", "121_aa"]).to_numpy()
y_test = dataset_test()[["121_aa"]].to_numpy()


model.accuracy(X_test, y_test)
