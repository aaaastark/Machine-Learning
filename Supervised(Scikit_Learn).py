# -------------------------------------------------------------------------------------------------------------------- #
#                        Loading the Data set
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.DESCR)
#output: .. _digits_dataset:
# Optical recognition of handwritten digits dataset
# --------------------------------------------------
# **Data Set Characteristics:**
#     :Number of Instances: 5620
#     :Number of Attributes: 64
#     :Attribute Information: 8x8 image of integer pixels in the range 0..16.
#     :Missing Attribute Values: None
#     :Creator: E. Alpaydin (alpaydin '@' boun.edu.tr)
#     :Date: July; 1998
#
# This is a copy of the test set of the UCI ML hand-written digits datasets
# https://archive.ics.uci.edu/ml/datasets/Optical+Recognition+of+Handwritten+Digits
#
# The data set contains images of hand-written digits: 10 classes whereeach class refers to a digit.
#
# Preprocessing programs made available by NIST were used to extractnormalized bitmaps of handwritten
# digits from a preprinted form. From a total of 43 people, 30 contributed to the training set and different 13
# to the test set. 32x32 bitmaps are divided into nonoverlapping blocks of4x4 and the number of on pixels are
# counted in each block. This generates an input matrix of 8x8 where each element is an integer in the range
# 0..16. This reduces dimensionality and gives in variance to small distortions.
# For info on NIST preprocessing routines, see M.D. Garris, J. L. Blue, G. T. Candela, D. L. Dimmick, J. Geist, P. J. Grot
# her, S. A. Janet, and C.L. Wilson, NIST Form-Based Handprint Recognition System, NISTIR 5469,1994.
# .. topic:: References
#   - C. Kaynak (1995) Methods of Combining Multiple Classifiers and Their
#     Applications to Handwritten Digit Recognition, MSc Thesis, Institute of
#     Graduate Studies in Science and Engineering, Bogazici University.
#   - E. Alpaydin, C. Kaynak (1998) Cascading Classifiers, Kybernetika.
#   - Ken Tang and Ponnuthurai N. Suganthan and Xi Yao and A. Kai Qin.
#     Linear dimensionalityreduction using relevance weighted LDA. School of
#     Electrical and Electronic Engineering Nanyang Technological University. 2005.
#   - Claudio Gentile. A New Approximate Maximal Margin Classificationlgorithm. NIPS. 2000.
# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#      Loading the Data set.And checking the Sample and Target Sizes, and Sample Digit Images
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.target[::100]) # analysis the target contains images label's
#output: [0 4 1 7 4 8 2 2 4 4 1 9 7 3 2 1 2 5]
print(digits.data.shape) # checking the number of samples(row) and features(columns). Row=1797,Columns=64
#output: (1797, 64)
print(digits.target.shape) # conform the number of target values matches the number of samples by looking images.
#output: (1797,)

# Here any number put now.
print(digits.images[13]) # two dimensional images pixels represents and scikit-learn stores into Numpy floating-points.
#output: [[ 0.  2.  9. 15. 14.  9.  3.  0.]
 # [ 0.  4. 13.  8.  9. 16.  8.  0.]
 # [ 0.  0.  0.  6. 14. 15.  3.  0.]
 # [ 0.  0.  0. 11. 14.  2.  0.  0.]
 # [ 0.  0.  0.  2. 15. 11.  0.  0.]
 # [ 0.  0.  0.  0.  2. 15.  4.  0.]
 # [ 0.  1.  5.  6. 13. 16.  6.  0.]
 # [ 0.  2. 12. 12. 13. 11.  0.  0.]]
print(digits.images[10]) # two dimensional images pixels represents and scikit-learn stores into Numpy floating-points.
#output: [[ 0.  0.  1.  9. 15. 11.  0.  0.]
 # [ 0.  0. 11. 16.  8. 14.  6.  0.]
 # [ 0.  2. 16. 10.  0.  9.  9.  0.]
 # [ 0.  1. 16.  4.  0.  8.  8.  0.]
 # [ 0.  4. 16.  4.  0.  8.  8.  0.]
 # [ 0.  1. 16.  5.  1. 11.  3.  0.]
 # [ 0.  0. 12. 12. 10. 10.  0.  0.]
 # [ 0.  0.  1. 10. 13.  3.  0.  0.]]

# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#         Loading the Data set.Preparing the Data for Use with Scikt-Learn
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
from sklearn.datasets import load_digits
digits = load_digits()
print(digits.data[10])
#output: [ 0.  0.  1.  9. 15. 11.  0.  0.  0.  0. 11. 16.  8. 14.  6.  0.  0.  2.
# 16. 10.  0.  9.  9.  0.  0.  1. 16.  4.  0.  8.  8.  0.  0.  4. 16.  4.
# 0.  8.  8.  0.  0.  1. 16.  5.  1. 11.  3.  0.  0.  0. 12. 12. 10. 10.
# 0.  0.  0.  0.  1. 10. 13.  3.  0.  0.]
# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#         Splitting the Data for Training and Testing. Loading the Data set from Scikit-learn
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
from  sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
digits = load_digits()
X_train,X_test,Y_train,Y_test = train_test_split(digits.data, digits.target, random_state=10)
print('The x-axis is training and testing modules',end='\n')
print('Train \n',X_train)
print('Test \n',X_test)
#output: The x-axis is training and testing modules
# Train
#  [[ 0.  0.  0. ...  0.  0.  0.]
#  [ 0.  0.  1. ... 15.  6.  0.]
#  [ 0.  0.  5. ...  0.  0.  0.]
#  ...
#  [ 0.  0.  6. ... 16. 16. 12.]
#  [ 0.  0. 15. ... 13.  6.  0.]
#  [ 0.  0.  2. ... 11. 16.  6.]]
# Test
#  [[ 0.  0.  8. ...  0.  0.  0.]
#  [ 0.  1. 13. ... 16. 16.  0.]
#  [ 0.  1. 13. ...  1.  0.  0.]
#  ...
#  [ 0.  0.  2. ...  3.  0.  0.]
#  [ 0.  0.  1. ...  7.  0.  0.]
#  [ 0.  0.  2. ... 16.  7.  0.]]
print('The y-axis is training and testing modules',end='\n')
print('Train \n',Y_train)
print('Test \n',Y_test)
#output: The y-axis is training and testing modules
# Train
#  [9 3 1 ... 1 8 2]
# Test
#  [5 2 5 4 8 2 4 3 3 0 8 7 0 1 8 6 9 7 9 7 1 8 6
#  7 8 8 5 3 5 9 3 3 7 3 4 1 9
#  2 5 4 2 1 0 9 2 3 6 1 9 4 4 9 8 4 8 5 9 7 8 0
# 4 5 8 4 7 9 0 7 1 3 9 3 3 8
#  0 7 3 6 5 2 0 8 8 0 1 1 2 8 8 8 2 6 3 4 7 9 8
# 2 9 2 5 0 8 0 4 8 8 0 6 7 3
#  3 9 1 5 4 6 0 8 8 1 1 7 9 9 5 2 3 3 9 7 6 2 5
# 4 3 3 7 6 7 2 7 4 9 5 1 9 4
#  6 1 1 1 4 0 4 9 1 2 3 5 0 3 4 1 5 4 9 3 5 6 4
# 0 8 6 7 0 9 9 4 7 3 5 2 0 6
#  7 5 3 9 7 1 3 2 8 3 3 1 7 1 1 1 7 1 6 7 6 9 5
# 2 3 5 2 9 5 4 8 2 9 1 5 0 2
#  3 9 0 2 0 2 1 0 5 0 6 4 2 1 9 0 9 0 6 9 4 4 9
# 7 5 6 1 8 7 0 8 6 2 0 1 2 3
#  8 4 4 3 5 7 9 7 2 0 2 0 9 2 8 6 3 6 0 6 6 6 7
# 1 6 1 7 6 0 6 3 7 4 6 2 8 0
#  8 4 7 3 3 0 0 2 3 9 7 4 6 7 9 7 6 0 5 6 2 7 1
# 0 5 1 6 4 7 2 5 1 4 6 6 5 0
#  2 9 8 7 9 6 7 0 8 3 5 9 4 1 5 5 4 7 3 9 2 7 3
# 3 6 6 3 2 1 9 8 3 0 8 7 0 4
#  2 1 1 2 9 8 5 1 7 9 8 7 5 4 2 5 5 4 2 4 6 5 0
# 8 2 0 6 6 3 6 5 3 0 9 7 1 6
#  7 4 7 3 2 5 2 1 2 6 8 0 1 9 7 6 9 9 2 9 1 0 9
# 9 8 3 6 1 1 3 0 6 8 3 2 0 3
#  4 5 5 8 8 6]
# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#         Creating the Model. Loading the Data set from Scikit-learn
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
digits = load_digits()
X_train,X_test,Y_train,Y_test = train_test_split(digits.data, digits.target, random_state=10)
KNC = KNeighborsClassifier()
name = KNC.fit(X= X_train, y= Y_train)
print(name)
#output: KNeighborsClassifier()
predicated_1 = KNC.predict(X= X_train)
print(predicated_1) # x-axis training model create
#output: [9 3 1 ... 1 8 2]
predicated_2 = KNC.predict(X= X_test)
print(predicated_2) # x-axis testing model create
#output: [5 2 5 4 8 2 4 3 3 0 8 7 0 1 8 6 9 7 9 7 1 8 6
# 7 8 8 5 3 5 9 3 3 7 3 4 1 9
#  2 5 4 2 1 0 9 2 3 6 1 9 4 4 9 8 4 8 5 9 7 1 0
# 4 5 8 4 7 9 0 7 1 3 9 3 3 8
#  0 7 3 6 5 2 0 8 8 0 1 1 2 8 8 8 2 6 3 4 7 9 8
# 2 9 2 5 0 8 0 4 8 8 0 6 7 3
#  3 9 1 5 4 6 0 8 8 1 1 7 9 9 5 2 3 3 8 7 6 2 5
# 4 3 3 7 6 7 2 7 4 9 5 1 9 4
#  6 1 1 1 4 0 8 9 1 2 3 5 0 3 4 1 5 4 9 3 5 6 4
# 0 8 6 7 0 9 9 4 7 3 5 2 0 6
#  7 5 3 5 7 1 3 2 8 3 3 1 7 1 1 1 7 1 6 7 6 9 5
# 2 3 5 2 9 5 4 8 2 9 1 5 0 2
#  3 9 0 2 0 2 1 0 5 0 6 4 2 1 9 0 9 0 6 9 4 4 9
# 7 5 6 1 3 7 0 8 6 2 0 1 2 3
#  8 4 4 3 5 7 9 7 2 0 2 0 9 2 8 6 3 6 0 6 6 6 7
# 1 6 1 7 6 0 6 3 7 4 6 2 8 0
#  8 4 7 3 3 0 0 2 3 9 7 4 6 7 9 7 6 0 5 6 2 7 1
# 0 5 5 6 4 7 2 5 1 4 6 6 5 0
#  2 9 8 7 9 6 7 0 8 3 5 9 4 1 5 5 4 7 3 9 2 7 3
# 3 6 6 3 2 1 9 8 3 0 8 7 0 4
#  2 1 1 2 9 8 5 1 7 9 8 7 5 4 2 5 5 4 2 4 6 5 0
# 8 2 0 6 6 3 6 5 3 0 9 7 1 6
#  7 4 7 3 2 5 2 1 2 6 8 0 1 9 7 6 9 9 2 9 1 0 9
# 9 8 3 6 1 1 3 0 6 8 3 2 0 3
#  4 5 5 8 8 6]
expected_1 = Y_train
print(expected_1) # y-axis training model create
#output: [9 3 1 ... 1 8 2]
expected_2 = Y_test
print(expected_2) # y-axis testing model create
#output: [5 2 5 4 8 2 4 3 3 0 8 7 0 1 8 6 9 7 9 7 1 8 6
# 7 8 8 5 3 5 9 3 3 7 3 4 1 9
#  2 5 4 2 1 0 9 2 3 6 1 9 4 4 9 8 4 8 5 9 7 8 0
# 4 5 8 4 7 9 0 7 1 3 9 3 3 8
#  0 7 3 6 5 2 0 8 8 0 1 1 2 8 8 8 2 6 3 4 7 9 8
# 2 9 2 5 0 8 0 4 8 8 0 6 7 3
#  3 9 1 5 4 6 0 8 8 1 1 7 9 9 5 2 3 3 9 7 6 2 5
# 4 3 3 7 6 7 2 7 4 9 5 1 9 4
#  6 1 1 1 4 0 4 9 1 2 3 5 0 3 4 1 5 4 9 3 5 6 4
# 0 8 6 7 0 9 9 4 7 3 5 2 0 6
#  7 5 3 9 7 1 3 2 8 3 3 1 7 1 1 1 7 1 6 7 6 9 5
# 2 3 5 2 9 5 4 8 2 9 1 5 0 2
#  3 9 0 2 0 2 1 0 5 0 6 4 2 1 9 0 9 0 6 9 4 4 9
# 7 5 6 1 8 7 0 8 6 2 0 1 2 3
#  8 4 4 3 5 7 9 7 2 0 2 0 9 2 8 6 3 6 0 6 6 6 7
# 1 6 1 7 6 0 6 3 7 4 6 2 8 0
#  8 4 7 3 3 0 0 2 3 9 7 4 6 7 9 7 6 0 5 6 2 7 1
# 0 5 1 6 4 7 2 5 1 4 6 6 5 0
#  2 9 8 7 9 6 7 0 8 3 5 9 4 1 5 5 4 7 3 9 2 7 3
# 3 6 6 3 2 1 9 8 3 0 8 7 0 4
#  2 1 1 2 9 8 5 1 7 9 8 7 5 4 2 5 5 4 2 4 6 5 0
# 8 2 0 6 6 3 6 5 3 0 9 7 1 6
#  7 4 7 3 2 5 2 1 2 6 8 0 1 9 7 6 9 9 2 9 1 0 9
# 9 8 3 6 1 1 3 0 6 8 3 2 0 3
#  4 5 5 8 8 6]

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
#         Estimator Method Score. Loading the Data set from Scikit-learn
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
digits = load_digits()
X_train,X_test,Y_train,Y_test = train_test_split(digits.data, digits.target, random_state=10)
KNC = KNeighborsClassifier()
name = KNC.fit(X= X_train, y= Y_train)
print(f'The score of model is {KNC.score(X_test,Y_test):.4%}')
#output: The score of model is 98.6667%

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
#         Confusion Matrix. Loading the Data set from Scikit-learn
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import  confusion_matrix
digits = load_digits()
X_train,X_test,Y_train,Y_test = train_test_split(digits.data, digits.target, random_state=10)
KNC = KNeighborsClassifier()
name = KNC.fit(X= X_train, y= Y_train)
predicated = KNC.predict(X= X_test)
expected = Y_test
confusion = confusion_matrix(y_true= expected, y_pred= predicated)
print(confusion)
#output: [[46  0  0  0  0  0  0  0  0  0]
 # [ 0 43  0  0  0  1  0  0  0  0]
 # [ 0  0 45  0  0  0  0  0  0  0]
 # [ 0  0  0 48  0  0  0  0  0  0]
 # [ 0  0  0  0 39  0  0  0  1  0]
 # [ 0  0  0  0  0 41  0  0  0  0]
 # [ 0  0  0  0  0  0 47  0  0  0]
 # [ 0  0  0  0  0  0  0 47  0  0]
 # [ 0  1  0  1  0  0  0  0 41  0]
 # [ 0  0  0  0  0  1  0  0  1 47]]

# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
#         Classification Report. Loading the Data set from Scikit-learn
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import  confusion_matrix
digits = load_digits()
X_train,X_test,Y_train,Y_test = train_test_split(digits.data, digits.target, random_state=10)
KNC = KNeighborsClassifier()
KNC.fit(X= X_train, y= Y_train)
predicated = KNC.predict(X= X_test)
expected = Y_test
from sklearn.metrics import classification_report
names = [str(digit) for digit in digits.target_names]
print(classification_report(expected,predicated,target_names=names))
#output:               precision    recall  f1-score   support
#            0       1.00      1.00      1.00
#     46
#            1       0.98      0.98      0.98
#     44
#            2       1.00      1.00      1.00
#     45
#            3       0.98      1.00      0.99
#     48
#            4       1.00      0.97      0.99
#     40
#            5       0.95      1.00      0.98
#     41
#            6       1.00      1.00      1.00
#     47
#            7       1.00      1.00      1.00
#     47
#            8       0.95      0.95      0.95
#     43
#            9       1.00      0.96      0.98
#     49
#
# accuracy                               0.99 450
# macro avg          0.99      0.99      0.99 450
# weighted avg       0.99      0.99      0.99 450
# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#         Visualizing the Confusion Matrix. Loading the Data set from Scikit-learn
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import  confusion_matrix
import pandas as pd
digits = load_digits()
X_train,X_test,Y_train,Y_test = train_test_split(digits.data, digits.target, random_state=10)
KNC = KNeighborsClassifier()
KNC.fit(X= X_train, y= Y_train)
predicated = KNC.predict(X= X_test)
expected = Y_test
confusion = confusion_matrix(y_true= expected, y_pred= predicated)
confusion_visulizig = pd.DataFrame(confusion, index= range(10), columns= range(10))
import seaborn as sns
axes = sns.heatmap(confusion_visulizig, annot= True, cmap= 'nipy_spectral_r')
# here create graph when your PC or Laptop is connected by Internet-Connection.
print(axes)
#output: AxesSubplot(0.125,0.11;0.62x0.77)

# -------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#         K-Fold Cross-Validation. Loading the Data set from Scikit-learn
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy
from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
digits = load_digits()
X_train,X_test,Y_train,Y_test = train_test_split(digits.data, digits.target, random_state=10)
KNC = KNeighborsClassifier()
KNC.fit(X= X_train, y= Y_train)
k_fold = KFold(n_splits= 10, random_state= 11, shuffle= True)
scores = cross_val_score(estimator= KNC, X= digits.data, y= digits.target, cv= k_fold)
print(scores)
#output: [0.97777778 0.99444444 0.98888889 0.97777778 0.98888889 0.99444444 0.97777778 0.98882682 1. 0.98324022]

# ------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#        Linear Regression the loading (data) into DataFram. Loading the Data set from Scikit-learn
import pandas as pd
name = pd.read_csv('D:\\rakahdon.csv')
print(name)
#output:    S.No    Name   Age      City    Salary
#   0        1     Tom      28    Toronto   20000
#   1        2     Lee      32   HongKong   30000
#   2        3    Steven    43    BayArea   83000
#   3        4     Ram      38  Hyderabad   39000
# ------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#        Linear Regression the loading (data) into DataFram. Loading the Data set from Scikit-learn
#        Training the Model
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy

from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.metrics import  confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
digits = load_digits()
X_train,X_test,Y_train,Y_test = train_test_split(digits.data, digits.target, random_state=10)
KNC = KNeighborsClassifier()
KNC.fit(X= X_train, y= Y_train)
predicated = KNC.predict(X= X_test)
expected = Y_test
confusion = confusion_matrix(y_true= expected, y_pred= predicated)
k_fold = KFold(n_splits= 10, random_state= 11, shuffle= True)
scores = cross_val_score(estimator= KNC, X= digits.data, y= digits.target, cv= k_fold)
linear_regression_name = LinearRegression()
name=linear_regression_name.fit(X=X_train,y=Y_train)
print(name)
#output: LinearRegression()
print(linear_regression_name.coef_)
#output: [-3.78492449e-16  1.55724373e-01 -4.54058951e-0
# 2 -3.94123882e-02
#   5.89438183e-02 -9.40083820e-03 -1.16853452e-0
# 2 -6.15634065e-03
#   1.08114604e+00 -4.34230863e-02  1.22387575e-0
# 1  3.65946933e-02
#  -7.18426197e-02 -6.68817828e-02  5.94945258e-0
# 2  2.28396370e-01
#   9.18041826e-01  9.80422801e-03  7.35576885e-0
# 2 -3.09470135e-02
#  -7.27954064e-02  4.29083818e-02 -4.21684563e-0
# 2 -2.30948755e-01
#  -7.68829445e-15 -1.71713832e-01  4.36080191e-0
# 2  8.56846801e-02
#   7.01296930e-02  9.63472231e-02 -3.58404344e-0
# 2 -2.65281272e+00
#  -2.19269047e-15 -1.62569010e-01 -2.25063576e-0
# 2  1.31919222e-01
#  -5.01085488e-02  2.95204223e-02  1.65758363e-0
# 3  2.68535194e-15
#   1.40349270e-01  1.02640646e-01 -1.39318665e-0
# 2 -3.53464123e-03
#   1.20907466e-01  4.43566675e-02  7.76455979e-0
# 3 -1.60346774e-02
#   6.13202719e-01  1.53601847e-02 -9.10468387e-0
# 3 -6.30717325e-02
#  -2.10994254e-01 -3.41718251e-02  1.05736374e-0
# 1 -1.41047192e-01
#  -1.32384381e+00 -1.33599567e-01  3.68863363e-0
# 2 -6.33435462e-02
#   5.61252032e-03 -8.32782912e-02 -2.11552956e-0
# 2 -1.65397635e-02]

print(linear_regression_name.intercept_)
#output: 3.68987827958686

# ------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#        Linear Regression the loading (data) into DataFram. Loading the Data set from Scikit-learn
#        Testing the Model
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy

from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import LinearRegression
digits = load_digits()
X_train,X_test,Y_train,Y_test = train_test_split(digits.data, digits.target, random_state=10)
KNC = KNeighborsClassifier()
KNC.fit(X= X_train, y= Y_train)
linear_regression_name = LinearRegression()
name=linear_regression_name.fit(X=X_train,y=Y_train)
predicated = linear_regression_name.predict(X_test)
expected = Y_test
print(name)
#output: LinearRegression()
for n1,n2 in zip(predicated[::5],expected[::5]):
    print(f'predicated:  {n1:.3f}, expected: {n2:.2f}')
#output: predicated:  3.699, expected: 5.00
# predicated:  0.916, expected: 2.00
# predicated:  8.238, expected: 8.00
# predicated:  6.123, expected: 6.00
# predicated:  1.888, expected: 1.00
# predicated:  8.967, expected: 8.00
# predicated:  6.029, expected: 3.00
# predicated:  2.940, expected: 1.00
# predicated:  2.499, expected: 2.00
# predicated:  6.008, expected: 3.00
# predicated:  4.311, expected: 4.00
# predicated:  5.993, expected: 5.00
# predicated:  5.151, expected: 4.00
# predicated:  7.188, expected: 9.00
# predicated:  6.898, expected: 9.00
# predicated:  6.584, expected: 7.00
# predicated:  2.185, expected: 0.00
# predicated:  3.878, expected: 1.00
# predicated:  2.979, expected: 2.00
# predicated:  4.472, expected: 9.00
# predicated:  4.625, expected: 5.00
# predicated:  8.169, expected: 8.00
# predicated:  1.532, expected: 3.00
# predicated:  5.729, expected: 4.00
# predicated:  3.850, expected: 1.00
# predicated:  4.485, expected: 5.00
# predicated:  7.321, expected: 7.00
# predicated:  5.219, expected: 3.00
# predicated:  2.027, expected: 2.00
# predicated:  3.777, expected: 1.00
# predicated:  4.826, expected: 1.00
# predicated:  8.836, expected: 9.00
# predicated:  1.321, expected: 0.00
# predicated:  2.037, expected: 4.00
# predicated:  4.576, expected: 4.00
# predicated:  -0.519, expected: 0.00
# predicated:  5.365, expected: 3.00
# predicated:  7.379, expected: 7.00
# predicated:  5.188, expected: 1.00
# predicated:  2.281, expected: 3.00
# predicated:  5.419, expected: 1.00
# predicated:  5.872, expected: 6.00
# predicated:  4.759, expected: 5.00
# predicated:  10.289, expected: 8.00
# predicated:  5.943, expected: 0.00
# predicated:  0.814, expected: 2.00
# predicated:  6.842, expected: 5.00
# predicated:  2.541, expected: 1.00
# predicated:  6.491, expected: 6.00
# predicated:  5.460, expected: 7.00
# predicated:  6.678, expected: 7.00
# predicated:  3.167, expected: 0.00
# predicated:  5.038, expected: 4.00
# predicated:  6.121, expected: 9.00
# predicated:  3.270, expected: 0.00
# predicated:  5.074, expected: 3.00
# predicated:  4.022, expected: 6.00
# predicated:  6.295, expected: 7.00
# predicated:  7.250, expected: 7.00
# predicated:  2.138, expected: 0.00
# predicated:  4.047, expected: 3.00
# predicated:  6.332, expected: 9.00
# predicated:  8.167, expected: 9.00
# predicated:  5.214, expected: 6.00
# predicated:  4.208, expected: 5.00
# predicated:  0.292, expected: 2.00
# predicated:  5.860, expected: 6.00
# predicated:  10.481, expected: 8.00
# predicated:  3.694, expected: 0.00
# predicated:  4.324, expected: 4.00
# predicated:  7.455, expected: 7.00
# predicated:  4.665, expected: 3.00
# predicated:  3.440, expected: 2.00
# predicated:  3.571, expected: 0.00
# predicated:  2.294, expected: 2.00
# predicated:  6.194, expected: 8.00
# predicated:  5.230, expected: 8.00
# predicated:  4.038, expected: 5.00
# predicated:  5.020, expected: 6.00
# predicated:  1.501, expected: 0.00
# predicated:  5.611, expected: 5.00
# predicated:  2.636, expected: 1.00
# predicated:  5.566, expected: 3.00
# predicated:  0.893, expected: 2.00
# predicated:  3.688, expected: 9.00
# predicated:  -0.506, expected: 2.00
# predicated:  4.747, expected: 9.00
# predicated:  2.446, expected: 1.00
# predicated:  4.588, expected: 3.00
# predicated:  6.860, expected: 5.00

# ------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#        Linear Regression the loading (data) into DataFram. Loading the Data set from Scikit-learn
#        Predicting Future Temperatures and Estimating Past Temperature
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy

from sklearn.neighbors import KNeighborsClassifier
from  sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits

from sklearn.linear_model import LinearRegression
digits = load_digits()
X_train,X_test,Y_train,Y_test = train_test_split(digits.data, digits.target, random_state=10)
KNC = KNeighborsClassifier()
KNC.fit(X= X_train, y= Y_train)
linear_regression_name = LinearRegression()
name=linear_regression_name.fit(X=X_train,y=Y_train)

print(name)
#output: LinearRegression()

predicated = (lambda x: linear_regression_name.coef_ * x + linear_regression_name.intercept_)
print(predicated(2020))
#output: [ 3.68987828e+00  3.18253112e+02 -8.80300298e+0
# 1 -7.59231459e+01
#   1.22756391e+02 -1.52998149e+01 -1.99145191e+0
# 1 -8.74592983e+00
#   2.18760487e+03 -8.40247561e+01  2.50912779e+0
# 2  7.76111588e+01
#  -1.41432213e+02 -1.31411323e+02  1.23868820e+0
# 2  4.65050546e+02
#   1.85813437e+03  2.34944189e+01  1.52276409e+0
# 2 -5.88230890e+01
#  -1.43356843e+02  9.03648095e+01 -8.14904034e+0
# 1 -4.62826607e+02
#   3.68987828e+00 -3.43172063e+02  9.17780768e+0
# 1  1.76772932e+02
#   1.45351858e+02  1.98311269e+02 -6.87077993e+0
# 1 -5.35499182e+03
#   3.68987828e+00 -3.24699522e+02 -4.17729641e+0
# 1  2.70166707e+02
#  -9.75293903e+01  6.33211314e+01  7.03819722e+0
# 0  3.68987828e+00
#   2.87195403e+02  2.11023984e+02 -2.44524920e+0
# 1 -3.45009701e+00
#   2.47922959e+02  9.32903467e+01  1.93742891e+0
# 1 -2.87001700e+01
#   1.24235937e+03  3.47174513e+01 -1.47015831e+0
# 1 -1.23715021e+02
#  -4.22518516e+02 -6.53372083e+01  2.17277354e+0
# 2 -2.81225450e+02
#  -2.67047461e+03 -2.66181247e+02  7.82002777e+0
# 1 -1.24264085e+02
#   1.50271693e+01 -1.64532270e+02 -3.90438189e+0
# 1 -2.97204439e+01]

# ------------------------------------------------------------------------------------------------------------------- #
