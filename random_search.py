
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import numpy as np
import category_encoders as ce
import matplotlib.pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100





df = pd.read_csv("BigDataV5.csv")

y = df['price']
x=df.drop('price', axis=1)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)



encoder = ce.JamesSteinEncoder(cols=['make', 'model', 'city','color','trans'])

X_train_tran=encoder.fit_transform(X_train,y_train)
X_test_tran=encoder.transform(X_test,y_test)


# Create the model
model = ensemble.GradientBoostingRegressor()

# Parameters we want to try
param_grid = {
    'n_estimators': np.arange(1000, 5000, 100),
    'max_depth': np.arange(5,10),
    'min_samples_leaf': np.arange(3,20),
    'learning_rate': np.arange(0.01,0.1),
    'max_features': np.arange(0.1,1),
    'loss': ['ls', 'lad', 'huber']
}


rs = RandomizedSearchCV(model, param_grid, n_jobs=-1, n_iter=1000)


rs.fit(X_train_tran, y_train)




# That is the combination that worked best.
print('Best params achieve a test score of', rs.score(X_test_tran, y_test), ':')

# Find the error rate on the training set using the best parameters
mse = mean_absolute_error(y_train, rs.predict(X_train_tran))
print("Training Set Mean Absolute Error: %.4f" % mse)

# Find the error rate on the test set using the best parameters
mse = mean_absolute_error(y_test, rs.predict(X_test_tran))
print("Test Set Mean Absolute Error: %.4f" % mse)

print(rs.best_params_)


r2= r2_score(y_test, model.predict(X_test_tran))

print("r2 score= %.4f" %r2)

mape= mean_absolute_percentage_error(y_test,model.predict(X_test_tran))
print('MAPE= %.4f' %mape)

plt.scatter(y_test,model.predict(X_test_tran))
plt.title('predictions')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.plot(y_test,y_test)
plt.show()
plt.boxplot(y_train)
plt.show()
