# -------------------------------------------------------------------------------------------------------------------- #
#         Visualizing the Data by creating the Diagram. Loading the Data set from Scikit-learn
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Write the code of 24-digits that displayed. Use the function subplots the 6-by-4 inch figure.
# The 6 is number of columns and 4 is number of rows.So, ncols=6 and nrows=4
from matplotlib import pyplot as plt
from sklearn.datasets import load_digits
digits = load_digits()
figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6,4))
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_title(target)
plt.tight_layout()
# -------------------------------------------------------------------------------------------------------------------- #
