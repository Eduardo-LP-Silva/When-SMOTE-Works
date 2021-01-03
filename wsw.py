import os
import csv
import argparse
import traceback
from numpy.lib.function_base import append
import pandas as pd
import numpy as np
import smote_variants as sv
from shutil import copyfile
from configparser import ConfigParser
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from pymfe.mfe import MFE

CONFIG_PATH = './config.ini'
DATA_DIR_PATH = './data/'
RESULTS_PATH = './results.csv'
RESULTS_BACKUP_PATH = './results_backup.csv'
MISSING_VALUES_THRESHOLD = 10
SMOTE_VARIANTS = [sv.SMOTE, sv.SMOTE_TomekLinks, sv.SMOTE_ENN, sv.Borderline_SMOTE1, sv.Borderline_SMOTE2, 
sv.LLE_SMOTE, sv.distance_SMOTE, sv.SMMO, sv.ADOMS, sv.Safe_Level_SMOTE, sv.MSMOTE, sv.DE_oversampling, 
sv.SVM_balance, sv.TRIM_SMOTE, sv.SMOTE_RSB, sv.ProWSyn, sv.SL_graph_SMOTE, sv.NRSBoundary_SMOTE, sv.SOI_CJ, 
sv.SMOTE_OUT, sv.SMOTE_Cosine, sv.Selected_SMOTE, sv.SMOTE_IPF, sv.SDSMOTE, sv.DSMOTE, sv.G_SMOTE, sv.Random_SMOTE, 
sv.VIS_RST, sv.NRAS, sv.SSO, sv.NDO_sampling, sv.Gaussian_SMOTE, sv.kmeans_SMOTE, sv.SN_SMOTE, sv.cluster_SMOTE]

MIN_MINORITY_NO = 10
UPSAMPLE_RATIOS = [0.5, 0.8, 1.0]
RESULTS_COLS = ['DATASET', 'SMOTE-VARIANT', 'UPSAMPLE-RATIO', 'AUC-IMB', 'AUC-BLC']

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

def train(clf, x, y, k=MIN_MINORITY_NO):
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
    mfe = MFE(groups=['general', 'statistical', 'info-theory', 'model-based', 'landmarking', 'clustering', 'concept', 
        'complexity'], random_state=42)
    mfe.fit(x, y, verbose=0, suppress_warnings=True)
    ft = mfe.extract(suppress_warnings=True)
    #print("\n".join("{:50} {:30}".format(x, y) for x, y in zip(ft[0], ft[1])))
    return ft

def multiclass_to_binary(y):
    classes = y.value_counts()

    if len(y.index) > 2:
        majority_no = classes.max()

        if majority_no < MIN_MINORITY_NO:
            raise Exception('Max number of examples of a class in Y is %d, must be >= %d' % (majority_no, MIN_MINORITY_NO))

        minority_no = classes.min()
        minority_class = classes.idxmin()

        while minority_no < MIN_MINORITY_NO:
            classes.drop(minority_class, inplace=True)
            minority_no = classes.min()
            minority_class = classes.idxmin()

        y = y.transform(lambda x: 1 if x == minority_class else 0)

    return y

def replace_missing_vals(x, formated):
    try:
        x = x.replace({'?': np.nan})
    except:
        pass
    
    if x.isna().values.any():
        missing_val_cols = x.loc[:, x.isna().any()].columns

        for col in missing_val_cols:
            if not formated:
                missing_vals_no = x[col].isna().sum()

                try:
                    x[col] = pd.to_numeric(x[col], downcast='float').values
                except:
                    pass

                if missing_vals_no > MISSING_VALUES_THRESHOLD: # Assume continuous attribute
                    x[col].fillna(x[col].mean(), inplace=True)
                else:
                    x[col].fillna(x[col].value_counts().idxmax(), inplace=True)
            else:
                if x[col].dtype == 'category':
                    x[col].fillna(x[col].value_counts().idxmax(), inplace=True)
                else:
                    x[col].fillna(x[col].mean(), inplace=True)

        if not formated:
            x = x.convert_dtypes()

    return x

def parse_dat_file(dataset_path):
    dataset_file = open(dataset_path, 'r')
    line = dataset_file.readline()

    if not line.startswith('@relation'):
        print('Unknown .dat annotation %s in %s dataset' % (line, dataset_path))
        return None, None

    attr_types = {}
    skip_rows = 1
    
    while True:
        line = dataset_file.readline()
        skip_rows = skip_rows + 1

        if line.startswith('@attribute'):
            line_split = line.split(' ')

            if line_split[2].startswith('{'):
                attr_types[skip_rows - 2] = 'category'
            elif line_split[2] == 'real':
                attr_types[skip_rows - 2] = 'float64'
            elif line_split[2] == 'integer':
                attr_types[skip_rows - 2] = 'int64'
            else:
                print('Unknown .dat attribute type %s in %s dataset' % (line_split[2], dataset_path))
                attr_types[skip_rows - 2] = 'object'

        elif line.startswith('@data'):
            return attr_types, skip_rows
        else:
            print('Unknown .dat annotation %s in %s dataset' % (line, dataset_path))

def load_data(dataset_path):
    attr_types = None
    skip_rows = None
    formated = False

    if dataset_path[-4:] == '.dat':
        attr_types, skip_rows = parse_dat_file(dataset_path)
        formated = True
        
    data = pd.read_csv(dataset_path, header=None, index_col=False, skiprows=skip_rows, dtype=attr_types)
    pred_attrs_len = len(data.columns) - 1
    x = data[data.columns[:pred_attrs_len]]
    y = data[data.columns[pred_attrs_len]]

    x = replace_missing_vals(x, formated)
    x = pd.get_dummies(x)

    y = multiclass_to_binary(y)

    return x,y

def run_experiment(results_file, results_writer, dataset_file_name, test_set_ratio=0.2):
        x, y = load_data(os.path.join(DATA_DIR_PATH, dataset_file_name))
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_set_ratio, stratify=y, random_state=42)
        mf = calc_mf(x_train.values, np.array(y_train.values))
        clf = train_random_forest(x_train, y_train)
        auc_imb = '{auc:.3f}'.format(auc=test(clf, x_test, y_test))
    
        for i in range(len(SMOTE_VARIANTS)):
            try:
                print('\n\n--> Applying %s' % SMOTE_VARIANTS[i].__name__)
                minority_ratio = calc_minority_ratio(y_train)
            except:
                traceback.print_exc()
                continue
            
            for upsample_ratio in UPSAMPLE_RATIOS:
                if minority_ratio < upsample_ratio:
                    try:
                        print('\n---> %.1f Upsample Ratio' % upsample_ratio)
                        smote_variant = SMOTE_VARIANTS[i](proportion=upsample_ratio, n_neighbors=3, random_state=42)
                        x_train_upsampled, y_train_upsampled = smote_variant.fit_resample(x_train.values, y_train.values)
                        
                        clf = train_random_forest(x_train_upsampled, y_train_upsampled)
                        auc_smote = '{auc:.3f}'.format(auc=test(clf, x_test, y_test))
                        mf_smote = calc_mf(x_train_upsampled, np.array(y_train_upsampled))
                        original_mf = mf[1].copy()

                        for j in range(len(mf_smote[1])):
                            original_mf[j] = '%.3f|%.3f' % (float(mf[1][j]), float(mf_smote[1][j]))

                        results_row = [dataset_file_name, SMOTE_VARIANTS[i].__name__, upsample_ratio, auc_imb, auc_smote]
                        results_row = results_row + original_mf

                        results_writer.writerow(results_row)
                        results_file.flush()
                    except:
                        traceback.print_exc()

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

        if config.has_option('main', 'files_read'):
            config.remove_option('main', 'files_read')

    files_read = []

    if config.has_option('main', 'files_read'):
        files_read = config.get('main', 'files_read').split(',')

    with open(RESULTS_PATH, open_mode, newline='') as results_file:
        results_writer = csv.writer(results_file, delimiter=',')

        if results_file.tell() == 0:
            rand_x = np.random.randint(1, 100, size=(50, 4))
            rand_y = np.random.randint(0, 2, size=(50, 1))
            mf = calc_mf(rand_x, rand_y)
            header = RESULTS_COLS.copy()
            header = header + mf[0]
            results_writer.writerow(header)

        data_dir = os.fsencode(DATA_DIR_PATH)
        data_dir_files = os.listdir(data_dir)
        dataset_file_name = ''

        for dataset_file in data_dir_files:
            dataset_file_name = os.fsdecode(dataset_file)

            if dataset_file_name in files_read:
                continue

            files_read.append(dataset_file_name)

            try:
                print('\n\n\n-> Processing %s dataset' % dataset_file_name)
                run_experiment(results_file, results_writer, dataset_file_name)
            except:
                traceback.print_exc()

        with open(CONFIG_PATH, 'w') as config_file:
            if not config.has_section('main'):
                config.add_section('main')

            config.set('main', 'files_read', ','.join(files_read))
            config.write(config_file)
                
if __name__ == '__main__':
    main()