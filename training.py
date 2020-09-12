from statistics import mean
import time
import category_encoders as ce
import pandas as pd
from sklearn import ensemble
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100




df = pd.read_csv("BigDataV5.csv")




x = df.drop('price', axis=1)
y = df['price'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0, shuffle=True)

encoder = ce.JamesSteinEncoder(cols=['make', 'model', 'city', 'color', 'trans'])
X_train_tran = encoder.fit_transform(X_train, y_train)
X_test_tran = encoder.transform(X_test, y_test)

model = ensemble.GradientBoostingRegressor(
    n_estimators=4000,
    learning_rate=0.01,
    max_depth=8,
    min_samples_leaf=3,
    max_features=0.3,
    loss='huber')


start_time = time.time()
model.fit(X_train_tran, y_train)
print("training time:  %s seconds" % (time.time() - start_time))
prd = model.predict(X_test_tran)







#test_error = []
#for i, x in enumerate(prd):
#    test_error.append(prd[i] - y_test[i])
#prd+=mean(test_error)


#static range
rangeV=6000
counter = 0
for i, x in enumerate(prd):
    if y_test[i]<prd[i]:
        if (prd[i]-y_test[i])<rangeV:
            counter+=1
    if y_test[i]>prd[i]:
        if (y_test[i]-prd[i])<rangeV:
            counter+=1





cnt = [0]
zero = [0]
for x in range(200000):
    cnt.append(x)
    zero.append(0)
plt.scatter(y_test, test_error)
plt.xlabel('true values')
plt.ylabel('error')
plt.title('Error vs True values')
plt.plot(cnt, zero)
plt.show()


mae = mean_absolute_error(y_train, model.predict(X_train_tran))
print("Training Set Mean Absolute Error: %.4f" % mae)

mae = mean_absolute_error(y_test, prd)
print("Test Set Mean Absolute Error: %.4f" % mae)

r2 = r2_score(y_test, prd)
print("r2 score= %.4f" % r2)

mape = mean_absolute_percentage_error(y_test, prd)
print('MAPE= %.4f' % mape)
print('range accuracy(',rangeV,') =%.2f' %(counter/np.size(prd)*100))

plt.boxplot(y)
plt.ylabel('price')
plt.title('car prices')
plt.show()
plt.hist(y)
plt.ylabel('price')
plt.title('car prices')
plt.show()

plt.scatter(y_test, prd)
plt.title('predictions')
plt.xlabel('true values')
plt.ylabel('predicted values')
plt.plot(y_test, y_test)
plt.show()
