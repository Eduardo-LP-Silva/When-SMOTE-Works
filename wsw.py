import os
import csv
import argparse
import traceback
import pandas as pd
import numpy as np
from shutil import copyfile
from configparser import ConfigParser
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
from pymfe.mfe import MFE

CONFIG_PATH = './config.ini'
DATA_DIR_PATH = './data/'
RESULTS_PATH = './results.csv'
RESULTS_BACKUP_PATH = './results_backup.csv'
MISSING_VALUES_THRESHOLD = 10
SMOTE_VARIANTS = [SMOTE] # TODO Complete
UPSAMPLE_RATIOS = [0.3, 0.6, 0.8, 1.0]
RESULTS_COLS = ['DATASET', 'AUC-IMB', 'SMOTE-VARIANT', 'UPSAMPLE-RATIO', 'AUC-BLC'] #TODO Add metafeatures

# Returns a model's AUC
def calc_auc(clf, x_test, y_test):
    prob_y = clf.predict_proba(x_test)
    prob_y = [p[1] for p in prob_y]

    return roc_auc_score(y_test, prob_y)

# Tests and evaluates a trained model given its training and test data sets
def test(clf, x_test, y_test):
    auc = calc_auc(clf, x_test, y_test)
    print('ROC AUC: %.2f' % auc)
    
    return auc

def train(clf, x, y, k=5):
    cv = RepeatedStratifiedKFold(n_splits=k, n_repeats=3, random_state=42)
    scores = cross_validate(clf, x, y, scoring=['roc_auc'], cv=cv, n_jobs=-1)

    print('CrossVal ROC AUC: %.2f' % np.mean(scores['test_roc_auc']))

    return clf.fit(x, y)

def train_random_forest(x_train, y_train):
    clf = RandomForestClassifier(max_features='sqrt', criterion='gini', min_samples_split=5, min_samples_leaf=2,
        max_depth=None, n_estimators=500, random_state=42)
    return train(clf, x_train, y_train)

def calc_minority_ratio(y):
    class_counts = y.value_counts() 

    return class_counts.min() / class_counts.max()

def calc_mf(x, y):
    mfe = MFE(groups=['general', 'statistical', 'info-theory', 'clustering', 'complexity'], random_state=42)
    mfe.fit(x, y)
    ft = mfe.extract()
    return ft

def multiclass_to_binary(y):
    classes = y.value_counts()

    if len(y.index) > 2:
        minority_class = classes.idxmin()
        y = y.transform(lambda x: 1 if x == minority_class else 0)

    return y

def replace_missing_vals(x):
    try:
        x = x.replace({'?': np.nan})
    except:
        pass
    
    if x.isna().values.any():
        missing_val_cols = x.loc[:, x.isna().any()].columns

        for col in missing_val_cols:
            try:
                x[col] = pd.to_numeric(x[col], downcast='float').values
            except:
                pass

            missing_vals_no = x[col].isna().sum().sum()

            if missing_vals_no > MISSING_VALUES_THRESHOLD: # Assume continuos attribute
                x[col].fillna(x[col].mean(), inplace=True)
            else:
                x[col].fillna(x[col].value_counts().idxmax(), inplace=True)

        x = x.convert_dtypes()

    return x


def load_data(dataset_path):
    data = pd.read_csv(dataset_path, header=None, index_col=False)
    pred_attrs_len = len(data.columns) - 1
    x = data[data.columns[:pred_attrs_len]]
    y = data[data.columns[pred_attrs_len]]

    x = replace_missing_vals(x)
    x = pd.get_dummies(x)

    y = multiclass_to_binary(y)

    return x,y

def run_experiment(results_writer, dataset_file_name, test_set_ratio=0.2):
        x, y = load_data(os.path.join(DATA_DIR_PATH, dataset_file_name))
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_set_ratio, stratify=y, random_state=42)
        calc_mf(x_train.values, y_train.values)
        clf = train_random_forest(x_train, y_train)
        auc_imb = '{auc:.3f}'.format(auc=test(clf, x_test, y_test))
    
        for i in range(len(SMOTE_VARIANTS)):
            minority_ratio = calc_minority_ratio(y_train)
            
            for upsample_ratio in UPSAMPLE_RATIOS:
                if minority_ratio < upsample_ratio:
                    smote_variant = SMOTE_VARIANTS[i](sampling_strategy=upsample_ratio, k_neighbors=3, random_state=42)
                    x_train_upsampled, y_train_upsampled = smote_variant.fit_resample(x_train.values, y_train.values)
                    
                    clf = train_random_forest(x_train_upsampled, y_train_upsampled)
                    auc_smote = '{auc:.3f}'.format(auc=test(clf, x_test, y_test))
                    #TODO Add metafeatures
                    results_writer.writerow([dataset_file_name, auc_imb, SMOTE_VARIANTS[i].__name__, upsample_ratio, auc_smote])

def main():
    config = ConfigParser()
    config.read(CONFIG_PATH)

    parser = argparse.ArgumentParser(description='When Smote Works')
    parser.add_argument('-c', dest='clear', action='store_true', default=False, help='Clear Result CSV')
    args = parser.parse_args()
    open_mode = 'a'

    if args.clear:
        open_mode = 'w'
        copyfile(RESULTS_PATH, RESULTS_BACKUP_PATH)
        config.remove_option('main', 'files_read')

    files_read = []

    if config.has_option('main', 'files_read'):
        files_read = config.get('main', 'files_read').split(',')

    with open(RESULTS_PATH, open_mode, newline='') as results_file:
        results_writer = csv.writer(results_file, delimiter=',')

        if results_file.tell() == 0:
            results_writer.writerow(RESULTS_COLS)

        data_dir = os.fsencode(DATA_DIR_PATH)
        data_dir_files = os.listdir(data_dir)
        dataset_file_name = ''

        for dataset_file in data_dir_files:
            dataset_file_name = os.fsdecode(dataset_file)

            if dataset_file_name in files_read:
                continue

            files_read.append(dataset_file_name)

            try:
                print('-> Processing %s dataset' % dataset_file_name)
                run_experiment(results_writer, dataset_file_name)
            except:
                traceback.print_exc()

        with open(CONFIG_PATH, 'w') as config_file:
            if not config.has_section('main'):
                config.add_section('main')

            config.set('main', 'files_read', ','.join(files_read))
            config.write(config_file)
                
if __name__ == '__main__':
    main()