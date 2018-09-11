import numpy as np
import pandas as pd
import itertools

from matplotlib import pyplot as plt


# Sklearn imports
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV, KFold, StratifiedKFold
from sklearn.base import BaseEstimator


# Taken from Sklearn docs
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')



def cleanData(df):
    # Convert commas to decimal points
    for col in ['v2', 'v3', 'v5', 'v6', 'v7']:
        df[col] = pd.to_numeric(df[col].str.replace(',', '.'))

    # Drop missing values from categorical columns v1 and v4 and the whole of v16
    df = df.dropna(subset=['v1', 'v4']).drop('v16', axis=1)

    # Drop v15 that is linearly dependent on v13
    df = df.drop(['v15'], axis=1)

    # v17 is the same is the output variable, so needs to be removed from the dataset
    df = df.drop('v17', axis=1)

    # # Impute missing values for the numerical columns
    # imr = Imputer(missing_values='NaN', strategy='mean', axis=0)
    # df[['v2', 'v13']] = imr.fit_transform(df[['v2', 'v13']])

    return df



def encodeCategoricals(df, dummyColList = [], labelCol = "", labelEncoding={}):
    df = pd.get_dummies(columns=dummyColList, data=df, drop_first=True)

    #Assuming labelEncoding has only 2 keys
    labelKeys = list(labelEncoding.keys())
    labelVals = list(labelEncoding.values())

    df[labelCol] = pd.to_numeric(df[labelCol].str.replace(labelKeys[0],
                                                               labelVals[0]).str.replace(labelKeys[1], labelVals[1]))

    return df


def equalizeColumns(df1, df2):
    df2Cols_add = set(df1.columns) - set(df2.columns)
    df1Cols_add = set(df2.columns) - set(df1.columns)

    for col in df1Cols_add:
        df1[col] = 0

    for col in df2Cols_add:
        df2[col] = 0

    # Make the order the same
    df2 = df2[df1.columns]

    return df1, df2


def gridSearch(gsPipe, param_grid, X, y, scoring='precision', random_state=0, cv=5):

    grid_search = GridSearchCV(estimator=gsPipe, param_grid=param_grid, n_jobs=-1,
                               scoring=scoring, cv=cv)

    grid_search.fit(X, y)
    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(param_grid[0].keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

    print("\n\nGrid scores:")
    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()



def logTransform(df, colList=[]):
    for col in colList:
        if col in df.columns:
            df.loc[df[col]>0, col] = np.log(df.loc[df[col]>0, col])

    return df


def load_clean_encode(fileName, delimiter=';'):
    dataSet = pd.read_csv(fileName, delimiter=delimiter)
    dataSet = cleanData(dataSet)
    dataSet = encodeCategoricals(dataSet,
                                  dummyColList=['v1', 'v4', 'v8', 'v9', 'v11', 'v12'],
                                  labelCol='classLabel',
                                  labelEncoding={'no.': '1', 'yes.': '0'})
    y = dataSet['classLabel']
    X = dataSet.drop('classLabel', axis=1)

    return X, y


def nested_cross_validation(estimator, param_grid, X, y, num_trials=1, n_splits=5):
    # Arrays to store scores
    non_nested_scores = np.zeros(num_trials)
    nested_scores = np.zeros(num_trials)

    print("Num Trials:", num_trials)

    # Loop for each trial
    for i in range(num_trials):
        print("Trial #%d" % i)
        # Choose cross-validation techniques for the inner and outer loops,
        # independently of the dataset.
        # E.g "LabelKFold", "LeaveOneOut", "LeaveOneLabelOut", etc.
        inner_cv = KFold(n_splits=n_splits, shuffle=True, random_state=i)
        outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=i)

        # Non_nested parameter search and scoring
        clf = cross_val_score(estimator=estimator, param_grid=param_grid, cv=inner_cv)
        clf.fit(X, y)
        non_nested_scores[i] = clf.best_score_
        print("\t Non-nested score: %f" %  non_nested_scores[i])
        # Nested CV with parameter optimization
        nested_score = cross_val_score(clf, X=X, y=y, cv=outer_cv)
        nested_scores[i] = nested_score.mean()
        print("\t Nested score: %f" % nested_scores[i])


    return non_nested_scores.mean(), nested_scores.mean()