import os
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter

DATA_DIR_PATH = './data/'
SMOTE_VARIANTS = [SMOTE] # TODO Complete
UPSAMPLE_RATIOS = [0.3, 0.6, 0.8, 1.0]

def calc_minority_ratio(y):
    minority_total = y[min(y, key=y.get)]

    return minority_total / len(y)

def run_experiment(dataset_path, test_set_ratio=0.2): 
    data = pd.read_csv(dataset_path, header=None, index_col=False)
    # handle missing values ??
    pred_attrs_len = len(data.columns) - 1
    x = data[data.columns[:pred_attrs_len]]
    y = data[data.columns[pred_attrs_len]]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_set_ratio, random_state=42)
    # calc metafeatures for x_train / y_train
    # apply classifier to dataset
 
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