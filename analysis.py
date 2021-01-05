import statistics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

FIGURES_FOLDER = './figures/'
DIFFICULTY_METAFEATURES = ['best_node.mean', 'elite_nn.mean', 'linear_discr.mean', 'naive_bayes.mean', 'one_nn.mean']
# Dataset Size -> nr_inst
# Attribute Number -> nr_attr
# Classification Difficulty -> Avg of best_node, elite_nn, linear_discr, naive_bayes, one_nn
# Metafeature Variability -> Difference between values
# Imbalance Ratio -> c2
def calc_result_metrics(results, datasets):
    dataset_sizes = []
    dataset_attrs_no = []
    dataset_difficulties = []
    dataset_mf_variabilities = []
    dataset_imb_ratios = []

    for dataset in datasets:
        dataset_results = results[results['DATASET'] == dataset]

        if len(dataset_results.index > 0):
            dataset_sizes.append(float(dataset_results.iloc[0]['nr_inst'].split('|')[0]))
            dataset_attrs_no.append(float(dataset_results.iloc[0]['nr_attr'].split('|')[0]))

            dif_mfs_values = []

            for dif_mf in DIFFICULTY_METAFEATURES:
                dif_mfs_values.append(float(dataset_results.iloc[0][dif_mf].split('|')[0]))

            dataset_difficulties.append(statistics.mean(dif_mfs_values))

            mfs_diffs = []

            for i in range(5, len(results.columns)):
                mfs_col_diffs = []

                for _, value in  dataset_results[dataset_results.columns[i]].iteritems():
                    mf_values = value.split('|')
                    mfs_col_diffs.append(float(mf_values[1]) - float(mf_values[0]))

                mfs_diffs.append(mfs_col_diffs)

            mfs_avgs = []

            for mf_diff_list in mfs_diffs:
                mfs_avgs.append(statistics.mean(mf_diff_list))

            dataset_mf_variabilities.append(mfs_avgs)
            dataset_imb_ratios.append(float(dataset_results.iloc[0]['c2'].split('|')[0]))

    '''
    print(dataset_sizes)
    print('\n')
    print(dataset_attrs_no)
    print('\n')
    print(dataset_difficulties)
    print('\n')
    print(len(dataset_mf_variabilities[0]))
    print('\n')
    print(dataset_imb_ratios)
    '''

    return dataset_sizes, dataset_attrs_no, dataset_difficulties, dataset_mf_variabilities, dataset_imb_ratios


def plot_avg_auc_dif(results, upsample_ratio):
    upsample_results = results[results['UPSAMPLE-RATIO'] == upsample_ratio]
    datasets = sorted(upsample_results['DATASET'].value_counts().index)

    bar_width = 0.75
    _, ax = plt.subplots(figsize=(20, 5))

    avg_diffs = []
    stdev_diffs = []
    auc_increased = {} # Calculate metrics
    auc_decreased = {} # Calculate metrics
    odd_cases = {}

    for dataset in datasets:
        smote_variant_rows = upsample_results[upsample_results['DATASET'] == dataset]
        auc_imb = smote_variant_rows.iloc[0]['AUC-IMB']
        diffs = []

        for _, row in smote_variant_rows.iterrows():
            diffs.append(row['AUC-BLC'] - auc_imb)

        avg_diff = round(statistics.mean(diffs), 2)
        stdev = round(statistics.stdev(diffs), 2)

        if avg_diff > 0:
            auc_increased[dataset] = avg_diff
        elif avg_diff <= 0:
            auc_decreased[dataset] = avg_diff

        for i in range(len(diffs)):
            if diffs[i] * avg_diff < 0 and abs(diffs[i]) > 0.03: # Opposite signs
                smote_variant = smote_variant_rows.iloc[i]['SMOTE-VARIANT']

                if smote_variant in odd_cases.keys():
                    odd_cases[smote_variant].append(dataset)
                else:
                    odd_cases[smote_variant] = [dataset]

        avg_diffs.append(avg_diff)
        stdev_diffs.append(stdev)

    print('\n-- Odd Cases --\n')
    print(odd_cases)

    title = 'Average AUC Difference (%d%% Minority-Majority Ratio)' % (upsample_ratio * 100)
    ax.bar(datasets, avg_diffs, bar_width, yerr=stdev_diffs)
    ax.set_title(title)

    plt.xticks(rotation=90)
    plt.ylim([-0.3, 0.3])
    plt.margins(0, 1)
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(FIGURES_FOLDER + title)
    plt.close()

    size_i, attr_no_i, diff_i, mf_var_i, imb_r_i = calc_result_metrics(upsample_results, auc_increased.keys())
    size_d, attr_no_d, diff_d, mf_var_d, imb_r_d = calc_result_metrics(upsample_results, auc_decreased.keys())

    metrics_df = pd.DataFrame(data={'Classification Improvement': True,
        'Average AUC Difference': auc_increased.values(), 'Dataset Size': size_i, 'Attributes': attr_no_i,
        'Average Classification Landmark': diff_i, 'Imbalance Ratio': imb_r_i})
    metrics_df_d = pd.DataFrame(data={'Classification Improvement': False,
        'Average AUC Difference': auc_decreased.values(), 'Dataset Size': size_d,
        'Attributes': attr_no_d, 'Average Classification Landmark': diff_d, 'Imbalance Ratio': imb_r_d})
    metrics_df = metrics_df.append(metrics_df_d)

    plot_scatter(metrics_df, 'Dataset Size', 'Average AUC Difference', 'Classification Improvement', 'AUC-Size-(%d%% Minority-Majority Ratio)' % (upsample_ratio * 100))
    plot_scatter(metrics_df, 'Attributes', 'Average AUC Difference', 'Classification Improvement', 'AUC-Attributes (%d%% Minority-Majority Ratio)' % (upsample_ratio * 100))
    plot_scatter(metrics_df, 'Average Classification Landmark', 'Average AUC Difference', 'Classification Improvement',
        'AUC-Difficulty (%d%% Minority-Majority Ratio)' % (upsample_ratio * 100))
    plot_scatter(metrics_df, 'Imbalance Ratio', 'Average AUC Difference', 'Classification Improvement',
        'AUC-IMBR (%d%% Minority-Majority Ratio)' % (upsample_ratio * 100))

    avg_metrics_i = [int(statistics.mean(size_i)), int(statistics.mean(attr_no_i)), round(statistics.mean(diff_i), 2), round(statistics.mean(imb_r_i), 2)]
    avg_metrics_d = [int(statistics.mean(size_d)), int(statistics.mean(attr_no_d)), round(statistics.mean(diff_d), 2), round(statistics.mean(imb_r_d), 2)]
    avg_metrics_all = [int(metrics_df['Dataset Size'].mean()), int(metrics_df['Attributes'].mean()), 
        round(metrics_df['Average Classification Landmark'].mean(), 2), round(metrics_df['Imbalance Ratio'].mean(), 2)]
    avg_metrics_df = pd.DataFrame(data={'Average Value (CI)': avg_metrics_i, 'Average Value (CD)': avg_metrics_d,
        'Average Value (All)': avg_metrics_all}, index=['Dataset Size', 'Attributes', 'Average Classification Landmark',
        'Imbalance Ratio'])
    avg_metrics_df.to_csv('./Average Metrics (%d%% Minority-Majority Ratio).csv' % (upsample_ratio * 100))
    calc_metafeatures_avgs(mf_var_i, mf_var_d, upsample_results.columns[5:], upsample_ratio)

def calc_metafeatures_avgs(mf_i, mf_d, names, upsample_ratio):
    rows = []

    mf_i_df = pd.DataFrame(data=mf_i, columns=names)
    mf_d_df = pd.DataFrame(data=mf_d, columns=names)
    mf_all_df = mf_i_df.append(mf_d_df)

    for row_name in names:
        rows.append([round(mf_i_df[row_name].mean(), 2), round(mf_d_df[row_name].mean(), 2), round(mf_all_df[row_name].mean(), 2)])

    mf_avg_df = pd.DataFrame(rows, 
        columns=['Average Difference (CI)', 'Average Difference (CD)', 'Average Difference (All)'],
        index=names)
    mf_avg_df.to_csv('./Average Metafeature Difference (%d%% Minority-Majority Ratio).csv' % (upsample_ratio * 100))

    '''
    for i in range(len(mf_i)) and range(len(mf_d)):
        rows.append([round(statistics.mean(mf_i[i]), 2), round(statistics.mean(mf_d[i]), 2), 
            round(statistics.mean(mf_i[i] + mf_d[i]), 2)])

    mf_avg_df = pd.DataFrame(rows, 
        columns=['Average Difference (CI)', 'Average Difference (CD)', 'Average Difference (All)'],
        index=names)

    mf_avg_df.to_csv('./Average Metafeature Difference (%d%% Minority-Majority Ratio).csv' % (upsample_ratio * 100))
    '''
    '''
    mf_i_df = pd.DataFrame(data=mf_i, columns=names)
    mf_d_df = pd.DataFrame(data=mf_d, columns=names)

    #print(mf_i)
    print(mf_i_df)
    '''

def plot_scatter(data, x, y, hue, title):
    axes = sns.scatterplot(data=data, x=x, y=y,
        hue=hue)
    fig = axes.get_figure()
    fig.savefig(FIGURES_FOLDER + title)
    plt.close()

def main():
    results = pd.read_csv('./results.csv', header=0, index_col=False)
    plot_avg_auc_dif(results, 0.5)
    plot_avg_auc_dif(results, 0.8)
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