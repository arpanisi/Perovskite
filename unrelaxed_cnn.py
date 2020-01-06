import numpy as np
import pandas as pd

cxs_names = np.load('data/unrelaxed_cxs_names.npy')
X = np.load('data/unrelaxed_cxs.npy')
y = np.load('data/unrelexed_energies.npy')

X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
y_max = np.max(y)

y /= y_max

from perovskite import RegressionModel as RM

model = RM()
model.compile_regression_model()

history, r_sq = model.fit(X, y, test_size=0.3)

y_true = y * y_max
y_pred_scaled = model.predict(X).reshape(len(y))
y_pred = y_pred_scaled * y_max
results = pd.DataFrame({'Names': cxs_names, 'Observed':y_true, 'Predicted': y_pred})

results.to_excel('Formation_Energy_results.xlsx')

from perovskite import Activation
img_tensor = X[0]

img_tensor = img_tensor.reshape(1, 32, 32, 1)

act = Activation(model=model)
hm = act.generate_heatmap(img_tensor)

r, _, _ = model.r_squared(X, y, split_size=0.3)
print(r)

model.save_model('formation')

import seaborn as sns

sns.jointplot('Observed', 'Predicted', data=results)