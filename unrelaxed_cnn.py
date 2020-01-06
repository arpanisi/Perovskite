import numpy as np

cxs_names = np.load('unrelaxed_cxs_names.npy')
X = np.load('unrelaxed_cxs.npy')
y = np.load('unrelexed_energies.npy')

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y_max = np.max(y)

y /= y_max

from perovskite import RegressionModel as RM

regression = RM()
regression.build_model()
regression.compile_regression_model()

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

history = regression.model.fit(X_train, y_train, verbose=1,
                                batch_size=1000, epochs=200,
                                validation_data=(X_test, y_test))

r, _, _ = regression.r_squared(X_test, y_test)
print(r)

import seaborn as sns
y_pred = regression.model.predict(X).reshape(len(X))

sns.jointplot(y, y_pred)