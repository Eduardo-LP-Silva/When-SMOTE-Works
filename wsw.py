import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE

DATA_DIR_PATH = './data/'
SMOTE_VARIANTS = [SMOTE] # TODO Complete
UPSAMPLE_RATIOS = [0.3, 0.6, 0.8, 1.0]

# Returns a model's AUC
def calc_auc(clf, x_test, y_test):
    prob_y = clf.predict_proba(x_test)
    prob_y = [p[1] for p in prob_y]

    return roc_auc_score(y_test, prob_y)

# Tests and evaluates a trained model given its training and test data sets
def test(clf, x_test, y_test):
    print('ROC AUC: %.2f' % calc_auc(clf, x_test, y_test))

def train(clf, x, y, k=5):
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=3, random_state=42)
    scores = cross_validate(clf, x, y, scoring=['roc_auc'], cv=cv, n_jobs=-1)

    print('CrossVal ROC AUC: %.2f' % np.mean(scores['test_roc_auc']))

    return clf.fit(x, y)

def train_random_forest(x_train, y_train):
    clf = RandomForestClassifier(max_features='sqrt', criterion='gini', min_samples_split=5, min_samples_leaf=2,
        max_depth=None, n_estimators=500, random_state=42)
    return train(clf, x_train, y_train, 10)

def calc_minority_ratio(y):
    print(y.value_counts().min())
    minority_total = y.value_counts().min()

    return minority_total / len(y)

def run_experiment(dataset_path, test_set_ratio=0.2): 
    data = pd.read_csv(dataset_path, header=None, index_col=False)
    pred_attrs_len = len(data.columns) - 1
    x = data[data.columns[:pred_attrs_len]]
    y = data[data.columns[pred_attrs_len]]
    x = pd.get_dummies(x)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_set_ratio, stratify=y, random_state=42)
    # calc metafeatures for x_train / y_train
    # apply classifier to dataset
    clf = train_random_forest(x_train, y_train)
    test(clf, x_test, y_test)
 
    for i in range(len(SMOTE_VARIANTS)):
        minority_ratio = calc_minority_ratio(y_train)
        #print(minority_ratio)

        for upsample_ratio in UPSAMPLE_RATIOS:
            if minority_ratio < upsample_ratio:
                smote_variant = SMOTE_VARIANTS[i](sampling_strategy=upsample_ratio, k_neighbors=3, random_state=42)
                x_train_upsampled, y_train_upsampled = smote_variant.fit_resample(x_train, y_train)
                # recalc meta features
                # apply classifier
                print(upsample_ratio)

def main():
    data_dir = os.fsencode(DATA_DIR_PATH)

    for dataset_file in os.listdir(data_dir):
        dataset_file_name = os.fsdecode(dataset_file)
        run_experiment(os.path.join(DATA_DIR_PATH, dataset_file_name))

if __name__ == '__main__':
    main()