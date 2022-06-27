import numpy as np
from pandas import read_csv

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)
# Split-out validation dataset
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]
print(len(dataset))
my_samples = []
for x in range(500):
    np.random.seed(42)
    shuffled_indices = np.random.permutation(len(dataset))
    x = np.random.choice(shuffled_indices, size=50, replace=True)
    selected_indices_len = len(x)
    test_indices = shuffled_indices[:selected_indices_len]
    my_samples = dataset.iloc[test_indices]


print(len(my_samples))
print(my_samples)
