# -------------------------------------------------------------------------------------------------------------------- #
#        (Unsupervised) Linear Regression the loading (data) into DataFram. Loading the Data set from Scikit-learn
#        Loading the Digits Dataset
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy
from sklearn.datasets import load_digits
digits = load_digits()
print(digits)
#output: {'data': array([[ 0.,  0.,  5., ...,  0.,  0.,0.],
#        [ 0.,  0.,  0., ..., 10.,  0.,  0.],
#        [ 0.,  0.,  0., ..., 16.,  9.,  0.],
#        ...,
#        [ 0.,  0.,  1., ...,  6.,  0.,  0.],
#        [ 0.,  0.,  2., ..., 12.,  0.,  0.],
#        [ 0.,  0., 10., ..., 12.,  1.,  0.]]), '
# target': array([0, 1, 2, ..., 8, 9, 8]), 'frame
# ': None, 'feature_names': ['pixel_0_0', 'pixel_
# 0_1', 'pixel_0_2', 'pixel_0_3', 'pixel_0_4', 'p
# ixel_0_5', 'pixel_0_6', 'pixel_0_7', 'pixel_1_0
# ', 'pixel_1_1', 'pixel_1_2', 'pixel_1_3', 'pixe
# l_1_4', 'pixel_1_5', 'pixel_1_6', 'pixel_1_7',
# 'pixel_2_0', 'pixel_2_1', 'pixel_2_2', 'pixel_2
# _3', 'pixel_2_4', 'pixel_2_5', 'pixel_2_6', 'pi
# xel_2_7', 'pixel_3_0', 'pixel_3_1', 'pixel_3_2'
# , 'pixel_3_3', 'pixel_3_4', 'pixel_3_5', 'pixel
# _3_6', 'pixel_3_7', 'pixel_4_0', 'pixel_4_1', '
# pixel_4_2', 'pixel_4_3', 'pixel_4_4', 'pixel_4_
# 5', 'pixel_4_6', 'pixel_4_7', 'pixel_5_0', 'pix
# el_5_1', 'pixel_5_2', 'pixel_5_3', 'pixel_5_4',
#  'pixel_5_5', 'pixel_5_6', 'pixel_5_7', 'pixel_
# 6_0', 'pixel_6_1', 'pixel_6_2', 'pixel_6_3', 'p
# ixel_6_4', 'pixel_6_5', 'pixel_6_6', 'pixel_6_7
# ', 'pixel_7_0', 'pixel_7_1', 'pixel_7_2', 'pixe
# l_7_3', 'pixel_7_4', 'pixel_7_5', 'pixel_7_6',
# 'pixel_7_7'], 'target_names': array([0, 1, 2, 3
# , 4, 5, 6, 7, 8, 9]), 'images': array([[[ 0.,
# 0.,  5., ...,  1.,  0.,  0.],
#         [ 0.,  0., 13., ..., 15.,  5.,  0.],
#         [ 0.,  3., 15., ..., 11.,  8.,  0.],
#         ...,
#         [ 0.,  4., 11., ..., 12.,  7.,  0.],
#         [ 0.,  2., 14., ..., 12.,  0.,  0.],
#         [ 0.,  0.,  6., ...,  0.,  0.,  0.]],
#
#        [[ 0.,  0.,  0., ...,  5.,  0.,  0.],
#         [ 0.,  0.,  0., ...,  9.,  0.,  0.],
#         [ 0.,  0.,  3., ...,  6.,  0.,  0.],
#         ...,
#         [ 0.,  0.,  1., ...,  6.,  0.,  0.],
#         [ 0.,  0.,  1., ...,  6.,  0.,  0.],
#         [ 0.,  0.,  0., ..., 10.,  0.,  0.]],
#
#        [[ 0.,  0.,  0., ..., 12.,  0.,  0.],
#         [ 0.,  0.,  3., ..., 14.,  0.,  0.],
#         [ 0.,  0.,  8., ..., 16.,  0.,  0.],
#         ...,
#         [ 0.,  9., 16., ...,  0.,  0.,  0.],
#         [ 0.,  3., 13., ..., 11.,  5.,  0.],
#         [ 0.,  0.,  0., ..., 16.,  9.,  0.]],
#
#        ...,
#
#        [[ 0.,  0.,  1., ...,  1.,  0.,  0.],
#         [ 0.,  0., 13., ...,  2.,  1.,  0.],
#         [ 0.,  0., 16., ..., 16.,  5.,  0.],
#         ...,
#         [ 0.,  0., 16., ..., 15.,  0.,  0.],
#         [ 0.,  0., 15., ..., 16.,  0.,  0.],
#         [ 0.,  0.,  2., ...,  6.,  0.,  0.]],
#
#        [[ 0.,  0.,  2., ...,  0.,  0.,  0.],
#         [ 0.,  0., 14., ..., 15.,  1.,  0.],
#         [ 0.,  4., 16., ..., 16.,  7.,  0.],
#         ...,
#         [ 0.,  0.,  0., ..., 16.,  2.,  0.],
#         [ 0.,  0.,  4., ..., 16.,  2.,  0.],
#         [ 0.,  0.,  5., ..., 12.,  0.,  0.]],
#
#        [[ 0.,  0., 10., ...,  1.,  0.,  0.],
#         [ 0.,  2., 16., ...,  1.,  0.,  0.],
#         [ 0.,  0., 15., ..., 15.,  0.,  0.],
#         ...,
#         [ 0.,  4., 16., ..., 16.,  6.,  0.],
#         [ 0.,  8., 16., ..., 16.,  8.,  0.],
#         [ 0.,  1.,  8., ..., 12.,  1.,  0.]]]),
#  'DESCR': ".. _digits_dataset:\n\nOptical recog
# nition of handwritten digits dataset\n---------
# -----------------------------------------\n\n**
# Data Set Characteristics:**\n\n    :Number of I
# nstances: 5620\n    :Number of Attributes: 64\n
#     :Attribute Information: 8x8 image of intege
# r pixels in the range 0..16.\n    :Missing Attr
# ibute Values: None\n    :Creator: E. Alpaydin (
# alpaydin '@' boun.edu.tr)\n    :Date: July; 199
# 8\n\nThis is a copy of the test set of the UCI
# ML hand-written digits datasets\nhttps://archiv
# e.ics.uci.edu/ml/datasets/Optical+Recognition+o
# f+Handwritten+Digits\n\nThe data set contains i
# mages of hand-written digits: 10 classes where\
# neach class refers to a digit.\n\nPreprocessing
#  programs made available by NIST were used to e
# xtract\nnormalized bitmaps of handwritten digit
# s from a preprinted form. From a\ntotal of 43 p
# eople, 30 contributed to the training set and d
# ifferent 13\nto the test set. 32x32 bitmaps are
#  divided into nonoverlapping blocks of\n4x4 and
#  the number of on pixels are counted in each bl
# ock. This generates\nan input matrix of 8x8 whe
# re each element is an integer in the range\n0..
# 16. This reduces dimensionality and gives invar
# iance to small\ndistortions.\n\nFor info on NIS
# T preprocessing routines, see M. D. Garris, J.
# L. Blue, G.\nT. Candela, D. L. Dimmick, J. Geis
# t, P. J. Grother, S. A. Janet, and C.\nL. Wilso
# n, NIST Form-Based Handprint Recognition System
# , NISTIR 5469,\n1994.\n\n.. topic:: References\
# n\n  - C. Kaynak (1995) Methods of Combining Mu
# ltiple Classifiers and Their\n    Applications
# to Handwritten Digit Recognition, MSc Thesis, I
# nstitute of\n    Graduate Studies in Science an
# d Engineering, Bogazici University.\n  - E. Alp
# aydin, C. Kaynak (1998) Cascading Classifiers,
# Kybernetika.\n  - Ken Tang and Ponnuthurai N. S
# uganthan and Xi Yao and A. Kai Qin.\n    Linear
#  dimensionalityreduction using relevance weight
# ed LDA. School of\n    Electrical and Electroni
# c Engineering Nanyang Technological University.
# \n    2005.\n  - Claudio Gentile. A New Approxi
# mate Maximal Margin Classification\n    Algorit
# hm. NIPS. 2000."}

# ------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#        (Unsupervised) Linear Regression the loading (data) into DataFram. Loading the Data set from Scikit-learn
# Creating a TSNE Estimator for Dimensionality Reduction. Transforming the Digits Dataset's Features into Two Dimensions
# Visualizing the Reduced Data
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
digits = load_digits()
tsne_name = TSNE(n_components= 2, random_state= 10)
reduced_data = tsne_name.fit_transform(digits.data)
print(reduced_data.shape)  # samples is 1797 (rows) and features is 2 (columns)
#output: (1797, 2)
figure_1 = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],c='black')
print(figure_1) # printing the value. but graph automatically show when your PC or Laptop is connected to Internet.
#output: <matplotlib.collections.PathCollection object at 0x0000015EF27453A0>
# ------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#        (Unsupervised) Linear Regression the loading (data) into DataFram. Loading the Data set from Scikit-learn
# Creating a TSNE Estimator for Dimensionality Reduction. Transforming the Digits Dataset's Features into Two Dimensions
# Visualizing the Reduced Data
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy
from sklearn.datasets import load_digits
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
digits = load_digits()
tsne_name = TSNE(n_components= 2, random_state= 10)
reduced_data = tsne_name.fit_transform(digits.data)
figure_1 = plt.scatter(reduced_data[:, 0], reduced_data[:, 1],c=digits.target, cmap=plt.cm.get_cmap('nipy_spectral_r',10))
colorbar = plt.colorbar(figure_1)
print(colorbar) # printing the value. but graph automatically show when your PC or Laptop is connected to Internet.
#output: <matplotlib.colorbar.Colorbar object at 0x000002018D9A4B50>

# ------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#        (Unsupervised) Linear Regression the loading (data) into DataFram. Loading the Data set from Scikit-learn
#         K - Means Clustering of IRIS Dataset. Loading the iris dataset
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy
# from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# digits = load_digits()
iris = load_iris()
print(iris.DESCR) # The Bunch's DESCR attribute indicates that there are 150 samples
# (Number of Instances), each with four features (Number of Attributes).

#output:  .. _iris_dataset:
#
# Iris plants dataset
# --------------------
#
# **Data Set Characteristics:**
#
#     :Number of Instances: 150 (50 in each of th
# ree classes)
#     :Number of Attributes: 4 numeric, predictiv
# e attributes and the class
#     :Attribute Information:
#         - sepal length in cm
#         - sepal width in cm
#         - petal length in cm
#         - petal width in cm
#         - class:
#                 - Iris-Setosa
#                 - Iris-Versicolour
#                 - Iris-Virginica
#
#     :Summary Statistics:
#
#     ============== ==== ==== ======= ===== ====
# ================
#                     Min  Max   Mean    SD   Cla
# ss Correlation
#     ============== ==== ==== ======= ===== ====
# ================
#     sepal length:   4.3  7.9   5.84   0.83    0
# .7826
#     sepal width:    2.0  4.4   3.05   0.43   -0
# .4194
#     petal length:   1.0  6.9   3.76   1.76    0
# .9490  (high!)
#     petal width:    0.1  2.5   1.20   0.76    0
# .9565  (high!)
#     ============== ==== ==== ======= ===== ====
# ================
#
#     :Missing Attribute Values: None
#     :Class Distribution: 33.3% for each of 3 cl
# asses.
#     :Creator: R.A. Fisher
#     :Donor: Michael Marshall (MARSHALL%PLU@io.a
# rc.nasa.gov)
#     :Date: July, 1988
#
# The famous Iris database, first used by Sir R.A
# . Fisher. The dataset is taken
# from Fisher's paper. Note that it's the same as
#  in R, but not as in the UCI
# Machine Learning Repository, which has two wron
# g data points.
#
# This is perhaps the best known database to be f
# ound in the
# pattern recognition literature.  Fisher's paper
#  is a classic in the field and
# is referenced frequently to this day.  (See Dud
# a & Hart, for example.)  The
# data set contains 3 classes of 50 instances eac
# h, where each class refers to a
# type of iris plant.  One class is linearly sepa
# rable from the other 2; the
# latter are NOT linearly separable from each oth
# er.
#
# .. topic:: References
#
#    - Fisher, R.A. "The use of multiple measurem
# ents in taxonomic problems"
#      Annual Eugenics, 7, Part II, 179-188 (1936
# ); also in "Contributions to
#      Mathematical Statistics" (John Wiley, NY,
# 1950).
#    - Duda, R.O., & Hart, P.E. (1973) Pattern Cl
# assification and Scene Analysis.
#      (Q327.D83) John Wiley & Sons.  ISBN 0-471-
# 22361-1.  See page 218.
#    - Dasarathy, B.V. (1980) "Nosing Around the
# Neighborhood: A New System
#      Structure and Classification Rule for Reco
# gnition in Partially Exposed
#      Environments".  IEEE Transactions on Patte
# rn Analysis and Machine
#      Intelligence, Vol. PAMI-2, No. 1, 67-71.
#    - Gates, G.W. (1972) "The Reduced Nearest Ne
# ighbor Rule".  IEEE Transactions
#      on Information Theory, May 1972, 431-433.
#    - See also: 1988 MLC Proceedings, 54-64.  Ch
# eeseman et al"s AUTOCLASS II
#      conceptual clustering system finds 3 class
# es in the data.
#    - Many, many more ...

# ------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#        (Unsupervised) Linear Regression the loading (data) into DataFram. Loading the Data set from Scikit-learn
# K - Means Clustering of IRIS Dataset. Loading the iris dataset and checking the Numbers of Samples,Features and Targets
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy
# from sklearn.datasets import load_digits
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
# digits = load_digits()
iris = load_iris()
print(iris.data.shape) # 150 = rows and 4 = columns
#output: (150, 4)
print(iris.target.shape)
#output: (150,)
print(iris.target_names)
#output: ['setosa' 'versicolor' 'virginica']
print(iris.feature_names)
#output: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']

# ------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#        (Unsupervised) Linear Regression the loading (data) into DataFram. Loading the Data set from Scikit-learn
# K - Means Clustering of IRIS Dataset. Loading the iris dataset and checking the Numbers of Samples,Features and Targets
# Exploring the IRIS Dataset: Descriptive Statistics with Pandas
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
pd.set_option('max_columns', 5)
pd.set_option('display.width',None)
iris_name = pd.DataFrame(iris.data, columns= iris.feature_names)
iris_name['species'] = [iris.target_names[i] for i in iris.target]
print(iris_name.head())
#output:    sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm) species
# 0                5.1               3.5
#        1.4               0.2  setosa
# 1                4.9               3.0
#        1.4               0.2  setosa
# 2                4.7               3.2
#        1.3               0.2  setosa
# 3                4.6               3.1
#        1.5               0.2  setosa
# 4                5.0               3.6
#        1.4               0.2  setosa
pd.set_option('precision',2)
print(iris_name.describe())
#output:        sepal length (cm)  sepal width (cm)  petal length (cm)  petal width (cm)
# count             150.00            150.00
#         150.00            150.00
# mean                5.84              3.06
#           3.76              1.20
# std                 0.83              0.44
#           1.77              0.76
# min                 4.30              2.00
#           1.00              0.10
# 25%                 5.10              2.80
#           1.60              0.30
# 50%                 5.80              3.00
#           4.35              1.30
# 75%                 6.40              3.30
#           5.10              1.80
# max                 7.90              4.40
#           6.90              2.50
print(iris_name['species'].describe())
#output: count            150
# unique             3
# top       versicolor
# freq              50
# Name: species, dtype: object

# ------------------------------------------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------------------------------------------- #
#        (Unsupervised) Linear Regression the loading (data) into DataFram. Loading the Data set from Scikit-learn
# K - Means Clustering of IRIS Dataset. Loading the iris dataset and checking the Numbers of Samples,Features and Targets
# Exploring the IRIS Dataset: Descriptive Statistics with Pandas
# Scikit-Learn machine learning algorithms require samples to be stored in a two dimensional array of floating point values.
# The two dimensional array-like collection, such as a list of lists or pandas DataFrame
# The load_digits function from the sklearn.datasets module returns a scikit-learn Bunch object containing the digits data
# and information about the Digits dataset is called METADATA.
# Here use the K-Nearest Neighbors Algorithm.
# Metrics for Model Accuracy

from sklearn.datasets import load_iris
import pandas as pd

iris = load_iris()
pd.set_option('max_columns', 5)
pd.set_option('display.width',None)
iris_name = pd.DataFrame(iris.data, columns= iris.feature_names)
iris_name['species'] = [iris.target_names[i] for i in iris.target]

import seaborn as sns
sns.set(font_scale= 1.1)
sns.set_style('whitegrid')
grid = sns.pairplot(data=iris_name,vars= iris_name.columns[0:4],hue='species')
print(grid) # printing the value. but graph automatically show when your PC or Laptop is connected to Internet.
#output: <seaborn.axisgrid.PairGrid object at 0x000001E83E2081F0>

# ------------------------------------------------------------------------------------------------------------------- #
