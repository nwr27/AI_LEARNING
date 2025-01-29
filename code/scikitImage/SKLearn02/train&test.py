from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

X = load_iris().data
y = load_iris().target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print(f"X train : {X_train.shape}")
print(f"X test : {X_test.shape}")
print(f"y train : {y_train.shape}")
print(f"y test : {y_test.shape}")
