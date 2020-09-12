from statistics import mean
import pandas as pd
import category_encoders as ce
import numpy as np
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import matplotlib.pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


df = pd.read_csv("BigDataV5.csv")

y = df['price']
x = df.drop('price', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

encoder = ce.JamesSteinEncoder(cols=['make', 'model', 'city', 'color', 'trans'])
X_train_tran = encoder.fit_transform(X_train, y_train)
X_test_tran = encoder.transform(X_test, y_test)

model = ensemble.GradientBoostingRegressor()

param_grid = {
    'n_estimators': [500, 1000, 3000],
    'max_depth': [4, 6],
    'min_samples_leaf': [3, 5, 9, 17],
    'learning_rate': [0.1, 0.05, 0.02, 0.01],
    'max_features': [1.0, 0.3, 0.1],
    'loss': ['ls', 'lad', 'huber']
}
bs = GridSearchCV(model, param_grid, n_jobs=-1)

bs.fit(X_train_tran, y_train)
model = bs.best_estimator_

print(bs.best_params_)

scores = cross_val_score(model, X_train_tran, y_train, cv=5)
print('mean score for K-folds CV= %.4f' % mean(scores))

mae = mean_absolute_error(y_train, model.predict(X_train_tran))
print("Training Set Mean Absolute Error: %.4f" % mae)

mae = mean_absolute_error(y_test, model.predict(X_test_tran))
print("Test Set Mean Absolute Error: %.4f" % mae)

r2 = r2_score(y_test, model.predict(X_test_tran))

print("r2 score= %.4f" % r2)

mape = mean_absolute_percentage_error(y_test, model.predict(X_test_tran))
print('MAPE= %.4f' % mape)

plt.scatter(y_test, model.predict(X_test_tran))
plt.title('predictions')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.plot(y_test, y_test)
plt.show()
plt.boxplot(y_train)
plt.show()
