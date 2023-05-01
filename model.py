# %%
import pandas as pd
from sklearn.metrics import accuracy_score   

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split



# %%
# Read original dataset


iris_df = pd.read_csv('./dataset/Iris.csv')
# iris_df.sample(frac=1, random_state=seed)

# %%
X = iris_df[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris_df[['Species']]

# %%
# split data into train and test sets
# 70% training and 30% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# %%
# create an instance of the random forest classifier
clf = RandomForestClassifier(n_estimators=100)

# %%
# train the classifier on the training data
clf.fit(X_train, y_train)
# predict on the test set
y_pred = clf.predict(X_test)

# %%
# calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', {accuracy})

# %%



