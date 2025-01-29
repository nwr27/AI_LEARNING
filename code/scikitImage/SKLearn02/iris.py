from sklearn.datasets import load_iris

iris = load_iris()
print(iris)
# print(iris.DESCR)

# print(iris.data)
# print(iris.data.shape)

# print(iris.target)
# print(iris.target.shape)

# print(iris.feature_names)
# print(iris.target_names)

iris_df = load_iris(as_frame=True)
print(iris_df.data)