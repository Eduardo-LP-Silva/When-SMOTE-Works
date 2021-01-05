import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIGURES_FOLDER = './figures/'

def plot_avg_auc_dif(results, upsample_ratio):
    upsample_results = results[results['UPSAMPLE-RATIO'] == upsample_ratio]
    datasets = sorted(upsample_results['DATASET'].value_counts().index)

    bar_positions = np.arange(len(datasets))
    bar_width = 0.75

    _, ax = plt.subplots(figsize=(20, 5))

    avg_diffs = []
    stdev_diffs = []

    for dataset in datasets:
        smote_variant_rows = upsample_results[upsample_results['DATASET'] == dataset]
        auc_imb = smote_variant_rows.iloc[0]['AUC-IMB']
        diffs = []

        for _, row in smote_variant_rows.iterrows():
            diffs.append(row['AUC-BLC'] - auc_imb)

        avg_diffs.append(round(statistics.mean(diffs), 2))
        stdev_diffs.append(round(statistics.stdev(diffs), 2))

    title = 'Average AUC Difference (%d%% Minority-Majority Ratio)' % (upsample_ratio * 100)
    ax.bar(datasets, avg_diffs, bar_width, yerr=stdev_diffs)
    ax.set_title(title)

    plt.xticks(rotation=90)
    plt.ylim([-0.3, 0.3])
    plt.margins(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(FIGURES_FOLDER + title)

def main():
    results = pd.read_csv('./results.csv', header=0, index_col=False)
    plot_avg_auc_dif(results, 1.0)

''' 
def plot_data_smote_auc_bar(results):
    smote_variants = results['SMOTE-VARIANT'].value_counts().index.values
    upsample_results = results[results['UPSAMPLE-RATIO'] == 0.5]

    bar_positions = np.arange(len(upsample_results['DATASET'].value_counts().index))
    bar_no = len(bar_positions)
    bar_width = 0.11

    fig, ax = plt.subplots(figsize=(20, 5))

    for i in range(len(smote_variants)):
        smote_var = smote_variants[i]
        values = upsample_results[upsample_results['SMOTE-VARIANT'] == smote_var]['AUC-IMB'].values
        value_no = len(values)

        if value_no < bar_no:
            values = np.append(values, np.zeros(bar_no - value_no))

        plt.bar([p + bar_width * i for p in bar_positions], values, bar_width, label=smote_var)


    ax.set_ylabel('AUC')
    ax.set_title('AUC by dataset and SMOTE variant')
    ax.set_xticks([p + 1.5 * bar_width for p in bar_positions])
    ax.set_xticklabels(upsample_results['DATASET'])

    plt.xticks(rotation=90)
    #plt.xlim(min(bar_positions) - bar_width, max(bar_positions) + bar_width * 4)
    #plt.ylim([0, 1])
    plt.legend(smote_variants, loc='upper left')
    plt.tight_layout()
    plt.savefig(FIGURES_FOLDER + 'AUC by dataset and SMOTE variant')
'''

if __name__ == '__main__':
    main()