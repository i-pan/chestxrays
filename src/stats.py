import pandas as pd 
import numpy as np 
import scipy.stats
import os
import re

from sklearn.metrics import roc_auc_score, f1_score

# 1. Load all of the results files 
RESULTS_DIR = '.'

d121_nih_on_nih0 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_nih_densenet121_split0.csv')).sort_values('pid') 
d121_nih_on_nih1 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_nih_densenet121_split1.csv')).sort_values('pid')  
d121_nih_on_nih2 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_nih_densenet121_split2.csv')).sort_values('pid')  
d121_nih_on_rih0 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_rih_densenet121_split0.csv')).sort_values('pid')  
d121_nih_on_rih1 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_rih_densenet121_split1.csv')).sort_values('pid')  
d121_nih_on_rih2 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_rih_densenet121_split2.csv')).sort_values('pid')  

mbv2_nih_on_nih0 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_nih_mobilenetv2_split0.csv')).sort_values('pid')  
mbv2_nih_on_nih1 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_nih_mobilenetv2_split1.csv')).sort_values('pid')   
mbv2_nih_on_nih2 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_nih_mobilenetv2_split2.csv')).sort_values('pid')  
mbv2_nih_on_rih0 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_rih_mobilenetv2_split0.csv')).sort_values('pid')  
mbv2_nih_on_rih1 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_rih_mobilenetv2_split1.csv')).sort_values('pid')  
mbv2_nih_on_rih2 = pd.read_csv(os.path.join(RESULTS_DIR, 'nih_on_rih_mobilenetv2_split2.csv')).sort_values('pid')  

d121_rih_on_nih0 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_nih_densenet121_split0.csv')).sort_values('pid') 
d121_rih_on_nih1 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_nih_densenet121_split1.csv')).sort_values('pid') 
d121_rih_on_nih2 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_nih_densenet121_split2.csv')).sort_values('pid') 
d121_rih_on_rih0 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_rih_densenet121_split0.csv')).sort_values('pid') 
d121_rih_on_rih1 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_rih_densenet121_split1.csv')).sort_values('pid') 
d121_rih_on_rih2 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_rih_densenet121_split2.csv')).sort_values('pid') 

mbv2_rih_on_nih0 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_nih_mobilenetv2_split0.csv')).sort_values('pid')   
mbv2_rih_on_nih1 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_nih_mobilenetv2_split1.csv')).sort_values('pid')  
mbv2_rih_on_nih2 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_nih_mobilenetv2_split2.csv')).sort_values('pid')  
mbv2_rih_on_rih0 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_rih_mobilenetv2_split0.csv')).sort_values('pid')  
mbv2_rih_on_rih1 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_rih_mobilenetv2_split1.csv')).sort_values('pid')  
mbv2_rih_on_rih2 = pd.read_csv(os.path.join(RESULTS_DIR, 'rih_on_rih_mobilenetv2_split2.csv')).sort_values('pid')  

# 2. Ensemble models 
# NIH on NIH 
columns_to_ensemble = list(d121_nih_on_nih0.columns)
columns_to_ensemble.remove('pid') 
for col in columns_to_ensemble: 
    if re.search('y_true', col): 
        columns_to_ensemble.remove(col) 

d121_nih_on_nih = d121_nih_on_nih0.copy() 
for col in columns_to_ensemble:
    d121_nih_on_nih[col] = np.mean((d121_nih_on_nih0[col], d121_nih_on_nih1[col], d121_nih_on_nih2[col]), axis=0)

mbv2_nih_on_nih = mbv2_nih_on_nih0.copy() 
for col in columns_to_ensemble:
    mbv2_nih_on_nih[col] = np.mean((mbv2_nih_on_nih0[col], mbv2_nih_on_nih1[col], mbv2_nih_on_nih2[col]), axis=0)

# NIH on RIH 
d121_nih_on_rih = d121_nih_on_rih0.copy() 
d121_nih_on_rih['y_pred'] = np.mean((d121_nih_on_rih0.y_pred, d121_nih_on_rih1.y_pred, d121_nih_on_rih2.y_pred), axis=0)

mbv2_nih_on_rih = mbv2_nih_on_rih0.copy()  
mbv2_nih_on_rih['y_pred'] = np.mean((mbv2_nih_on_rih0.y_pred, mbv2_nih_on_rih1.y_pred, mbv2_nih_on_rih2.y_pred), axis=0)

# RIH on RIH 
d121_rih_on_nih = d121_rih_on_nih0.copy()  
d121_rih_on_nih['y_pred'] = np.mean((d121_rih_on_nih0.y_pred, d121_rih_on_nih1.y_pred, d121_rih_on_nih2.y_pred), axis=0)

mbv2_rih_on_nih = mbv2_rih_on_nih0.copy() 
mbv2_rih_on_nih['y_pred'] = np.mean((mbv2_rih_on_nih0.y_pred, mbv2_rih_on_nih1.y_pred, mbv2_rih_on_nih2.y_pred), axis=0)

d121_rih_on_rih = d121_rih_on_rih0.copy()  
d121_rih_on_rih['y_pred'] = np.mean((d121_rih_on_rih0.y_pred, d121_rih_on_rih1.y_pred, d121_rih_on_rih2.y_pred), axis=0)

mbv2_rih_on_rih = mbv2_rih_on_rih0.copy() 
mbv2_rih_on_rih['y_pred'] = np.mean((mbv2_rih_on_rih0.y_pred, mbv2_rih_on_rih1.y_pred, mbv2_rih_on_rih2.y_pred), axis=0)

d121_nih_on_nih['y_true'] = d121_nih_on_nih['y_true_Finding']
d121_nih_on_nih['y_pred'] = d121_nih_on_nih['y_pred_Finding']
mbv2_nih_on_nih['y_true'] = mbv2_nih_on_nih['y_true_Finding']
mbv2_nih_on_nih['y_pred'] = mbv2_nih_on_nih['y_pred_Finding']

d121_nih_on_rih = d121_nih_on_rih.groupby('pid').mean().reset_index()
mbv2_nih_on_rih = mbv2_nih_on_rih.groupby('pid').mean().reset_index()
d121_rih_on_rih = d121_rih_on_rih.groupby('pid').mean().reset_index()
mbv2_rih_on_rih = mbv2_rih_on_rih.groupby('pid').mean().reset_index()

# Exclude IDs in RIH train-val from RIH test
exclude_rih = pd.read_table(os.path.join(RESULTS_DIR, 'exclude_from_rih_test.txt'))
d121_nih_on_rih = d121_nih_on_rih[~d121_nih_on_rih.pid.isin(exclude_rih.iloc[:,0])].reset_index()
mbv2_nih_on_rih = mbv2_nih_on_rih[~mbv2_nih_on_rih.pid.isin(exclude_rih.iloc[:,0])].reset_index()
d121_rih_on_rih = d121_rih_on_rih[~d121_rih_on_rih.pid.isin(exclude_rih.iloc[:,0])].reset_index()
mbv2_rih_on_rih = mbv2_rih_on_rih[~mbv2_rih_on_rih.pid.isin(exclude_rih.iloc[:,0])].reset_index()


# 3. For NIH-trained models, compute AUROCs on NIH test data
findings = ['Hernia', 'Pneumonia', 'Fibrosis', 'Edema', 'Emphysema', 'Cardiomegaly', 'Pleural_Thickening', 'Consolidation', 'Pneumothorax', 'Mass', 'Nodule', 'Atelectasis', 'Effusion', 'Infiltration', 'Finding']
for find in findings: 
    d121_auroc = roc_auc_score(d121_nih_on_nih['y_true_{}'.format(find)], d121_nih_on_nih['y_pred_{}'.format(find)])
    mbv2_auroc = roc_auc_score(mbv2_nih_on_nih['y_true_{}'.format(find)], mbv2_nih_on_nih['y_pred_{}'.format(find)])
    print ('NIH-DenseNet121 [{finding}] : AUROC = {auc:.3f}'.format(finding=find, auc=d121_auroc))
    print ('NIH-MobileNetV2 [{finding}] : AUROC = {auc:.3f}'.format(finding=find, auc=mbv2_auroc))

# 4. For NIH-trained models, compute AUROCs on RIH test data 
print ('NIH-DenseNet121 [RIH] : AUROC = {auc:.3f}'.format(auc=roc_auc_score(d121_nih_on_rih.y_true, d121_nih_on_rih.y_pred)))
print ('NIH-MobileNetV2 [RIH] : AUROC = {auc:.3f}'.format(auc=roc_auc_score(mbv2_nih_on_rih.y_true, mbv2_nih_on_rih.y_pred)))

# 5. For RIH-trained models, compute AUROCs on NIH test data
print ('RIH-DenseNet121 [NIH] : AUROC = {auc:.3f}'.format(auc=roc_auc_score(d121_rih_on_nih.y_true, d121_rih_on_nih.y_pred)))
print ('RIH-MobileNetV2 [NIH] : AUROC = {auc:.3f}'.format(auc=roc_auc_score(mbv2_rih_on_nih.y_true, mbv2_rih_on_nih.y_pred)))

# 6. For RIH-trained models, compute AUROCs on RIH test data
print ('RIH-DenseNet121 [RIH] : AUROC = {auc:.3f}'.format(auc=roc_auc_score(d121_rih_on_rih.y_true, d121_rih_on_rih.y_pred)))
print ('RIH-MobileNetV2 [RIH] : AUROC = {auc:.3f}'.format(auc=roc_auc_score(mbv2_rih_on_rih.y_true, mbv2_rih_on_rih.y_pred)))

# 7. Compute 95% bootstrap CIs for % differences for each of the 14 findings
np.random.seed(88) 
num_bootstraps = 1000
for find in findings:
    differences = [] 
    for boot in xrange(num_bootstraps): 
        sample_indices = np.random.choice(range(len(d121_nih_on_nih)), len(d121_nih_on_nih), replace=True)
        d121_auroc = roc_auc_score(d121_nih_on_nih.loc[sample_indices, 'y_true_{}'.format(find)], d121_nih_on_nih.loc[sample_indices, 'y_pred_{}'.format(find)])
        mbv2_auroc = roc_auc_score(mbv2_nih_on_nih.loc[sample_indices, 'y_true_{}'.format(find)], mbv2_nih_on_nih.loc[sample_indices, 'y_pred_{}'.format(find)])
        differences.append(d121_auroc - mbv2_auroc)
    print ('Mean difference (95% CI) {finding} : {mean:.3f} ({lower:.3f}, {upper:.3f})'.format(finding=find, mean=np.mean(differences), lower=np.percentile(differences, 2.5), upper=np.percentile(differences, 97.5)))

# 8. Compute 95% bootstrap CIs for % differences for abnormal vs. normal on RIH and NIH test sets
def bootstrap_abnormal_vs_normal(df1, df2, seed=88, num_bootstraps=1000): 
    np.random.seed(seed) 
    differences = [] 
    for boot in xrange(num_bootstraps): 
        sample_indices = np.random.choice(range(len(df1)), len(df1), replace=True)
        d121_auroc = roc_auc_score(df1.loc[sample_indices, 'y_true'], df1.loc[sample_indices, 'y_pred'])
        mbv2_auroc = roc_auc_score(df2.loc[sample_indices, 'y_true'], df2.loc[sample_indices, 'y_pred'])
        differences.append(d121_auroc - mbv2_auroc)
    print ('Mean difference (95% CI) {finding} : {mean:.3f} ({lower:.3f}, {upper:.3f})'.format(finding=find, mean=np.mean(differences), lower=np.percentile(differences, 2.5), upper=np.percentile(differences, 97.5)))

# DenseNet121 : NIH vs RIH on NIH 
bootstrap_abnormal_vs_normal(d121_nih_on_nih, d121_rih_on_nih)
# MobileNetV2 : NIH vs RIH on NIH 
bootstrap_abnormal_vs_normal(mbv2_nih_on_nih, mbv2_rih_on_nih)
# DenseNet121 : NIH vs RIH on RIH 
bootstrap_abnormal_vs_normal(d121_nih_on_rih, d121_rih_on_rih)
# MobileNetV2 : NIH vs RIH on RIH 
bootstrap_abnormal_vs_normal(mbv2_nih_on_rih, mbv2_rih_on_rih)

# 9. Calculate correlation for NIH vs. RIH models on NIH/RIH test sets 
scipy.stats.pearsonr(d121_nih_on_nih.y_pred, d121_rih_on_nih.y_pred)
scipy.stats.pearsonr(mbv2_nih_on_nih.y_pred, mbv2_rih_on_nih.y_pred)
scipy.stats.pearsonr(d121_nih_on_rih.y_pred, d121_rih_on_rih.y_pred)
scipy.stats.pearsonr(mbv2_nih_on_rih.y_pred, mbv2_rih_on_rih.y_pred)

# 10. Calculate median difference and IQR in scores for NIH vs. RIH models on NIH/RIH test sets 
def calculate_median_diff_iqr(x, y): 
    med = np.median(x - y) 
    upper = np.percentile(x - y, 75)
    lower = np.percentile(x - y, 25)
    print ('Median difference (IQR) : {median:.3f} ({lower:.3f}, {upper:.3f})'.format(median=med, lower=lower, upper=upper))

def calculate_median_iqr(x): 
    med = np.median(x) 
    upper = np.percentile(x, 75)
    lower = np.percentile(x, 25)
    print ('Median (IQR) : {median:.3f} ({lower:.3f}, {upper:.3f})'.format(median=med, lower=lower, upper=upper))

calculate_median_iqr(d121_nih_on_nih.y_pred_Finding) ; calculate_median_iqr(d121_rih_on_nih.y_pred)
calculate_median_diff_iqr(d121_nih_on_nih.y_pred_Finding, d121_rih_on_nih.y_pred)
calculate_median_iqr(mbv2_nih_on_nih.y_pred_Finding) ; calculate_median_iqr(mbv2_rih_on_nih.y_pred)
calculate_median_diff_iqr(mbv2_nih_on_nih.y_pred_Finding, mbv2_rih_on_nih.y_pred)
calculate_median_iqr(d121_nih_on_rih.y_pred) ; calculate_median_iqr(d121_rih_on_rih.y_pred)
calculate_median_diff_iqr(d121_nih_on_rih.y_pred, d121_rih_on_rih.y_pred)
calculate_median_iqr(mbv2_nih_on_rih.y_pred) ; calculate_median_iqr(mbv2_rih_on_rih.y_pred)
calculate_median_diff_iqr(mbv2_nih_on_rih.y_pred, mbv2_rih_on_rih.y_pred)

# Make sure these are all positive
calculate_median_diff_iqr(d121_nih_on_nih.y_pred_Finding[d121_nih_on_nih.y_true_Finding == 1], d121_rih_on_nih.y_pred[d121_nih_on_nih.y_true_Finding == 1])
calculate_median_diff_iqr(mbv2_nih_on_nih.y_pred_Finding[mbv2_nih_on_nih.y_true_Finding == 1], mbv2_rih_on_nih.y_pred[mbv2_nih_on_nih.y_true_Finding == 1])
calculate_median_diff_iqr(d121_nih_on_rih.y_pred[d121_nih_on_rih.y_true == 1], d121_rih_on_rih.y_pred[d121_nih_on_rih.y_true == 1])
calculate_median_diff_iqr(mbv2_nih_on_rih.y_pred[mbv2_nih_on_rih.y_true == 1], mbv2_rih_on_rih.y_pred[mbv2_nih_on_rih.y_true == 1])
calculate_median_diff_iqr(d121_nih_on_nih.y_pred_Finding[d121_nih_on_nih.y_true_Finding == 0], d121_rih_on_nih.y_pred[d121_nih_on_nih.y_true_Finding == 0])
calculate_median_diff_iqr(mbv2_nih_on_nih.y_pred_Finding[mbv2_nih_on_nih.y_true_Finding == 0], mbv2_rih_on_nih.y_pred[mbv2_nih_on_nih.y_true_Finding == 0])
calculate_median_diff_iqr(d121_nih_on_rih.y_pred[d121_nih_on_rih.y_true == 0], d121_rih_on_rih.y_pred[d121_nih_on_rih.y_true == 0])
calculate_median_diff_iqr(mbv2_nih_on_rih.y_pred[mbv2_nih_on_rih.y_true == 0], mbv2_rih_on_rih.y_pred[mbv2_nih_on_rih.y_true == 0])


# 11. Look at examples of highest mismatch
def get_top_mismatches(nih_df, rih_df, k=5): 
    df = nih_df.merge(rih_df, on='pid', suffixes=('_nih', '_rih'))
    df['y_diff'] = np.abs(df.y_pred_nih - df.y_pred_rih)
    return df[['pid','y_true_nih','y_true_rih','y_pred_rih','y_pred_nih','y_diff']].sort_values('y_diff', ascending=False).head(n=k)

get_top_mismatches(d121_nih_on_nih, d121_rih_on_nih)
# 00020393_001.png / RIH 0.13 / NIH 0.97 / diff 0.83 
get_top_mismatches(mbv2_nih_on_nih, mbv2_rih_on_nih)
# 00015023_002.png / RIH 0.09 / NIH 0.92 / diff 0.84 
get_top_mismatches(d121_nih_on_rih, d121_rih_on_rih)
# aec57da76885b88efe1149e3e0999095 / RIH 0.10 / NIH 0.89 / diff 0.79 
get_top_mismatches(mbv2_nih_on_rih, mbv2_rih_on_rih)
# 383e4c365ed1d65a86edb98327483195 / RIH 0.06 / NIH 0.86 / diff 0.80 

# Calculate % positives in top 100 mismatches
get_top_mismatches(d121_nih_on_nih, d121_rih_on_nih, k=100).y_true_nih.mean()
get_top_mismatches(mbv2_nih_on_nih, mbv2_rih_on_nih, k=100).y_true_nih.mean()
get_top_mismatches(d121_nih_on_rih, d121_rih_on_rih, k=100).y_true_nih.mean()
get_top_mismatches(mbv2_nih_on_rih, mbv2_rih_on_rih, k=100).y_true_nih.mean()

# 12. Calculate F1 score at different thresholds
def get_f1_score_range(y_true, y_pred, thresholds=np.linspace(0.05, 0.95, 19)):
    return pd.DataFrame({'f1': [f1_score(y_true, y_pred > thres) for thres in thresholds],
                         'threshold': thresholds})

d121_nih_on_nih_f1 = get_f1_score_range(d121_nih_on_nih.y_true_Finding, d121_nih_on_nih.y_pred_Finding)
d121_nih_on_nih_f1['model'] = 'd121_on_nih'
mbv2_nih_on_nih_f1 = get_f1_score_range(mbv2_nih_on_nih.y_true_Finding, mbv2_nih_on_nih.y_pred_Finding)
mbv2_nih_on_nih_f1['model'] = 'mbv2_on_nih'
d121_nih_on_rih_f1 = get_f1_score_range(d121_nih_on_rih.y_true, d121_nih_on_rih.y_pred)
d121_nih_on_rih_f1['model'] = 'd121_on_rih'
mbv2_nih_on_rih_f1 = get_f1_score_range(mbv2_nih_on_rih.y_true, mbv2_nih_on_rih.y_pred)
mbv2_nih_on_rih_f1['model'] = 'mbv2_on_rih'

d121_rih_on_nih_f1 = get_f1_score_range(d121_rih_on_nih.y_true, d121_rih_on_nih.y_pred)
d121_rih_on_nih_f1['model'] = 'd121_on_nih'
mbv2_rih_on_nih_f1 = get_f1_score_range(mbv2_rih_on_nih.y_true, mbv2_rih_on_nih.y_pred)
mbv2_rih_on_nih_f1['model'] = 'mbv2_on_nih'
d121_rih_on_rih_f1 = get_f1_score_range(d121_rih_on_rih.y_true, d121_rih_on_rih.y_pred)
d121_rih_on_rih_f1['model'] = 'd121_on_rih'
mbv2_rih_on_rih_f1 = get_f1_score_range(mbv2_rih_on_rih.y_true, mbv2_rih_on_rih.y_pred)
mbv2_rih_on_rih_f1['model'] = 'mbv2_on_rih'

# Highest threshold/scores 
mbv2_nih_on_nih_f1.sort_values('f1', ascending=False).head()
d121_nih_on_nih_f1.sort_values('f1', ascending=False).head()
mbv2_rih_on_nih_f1.sort_values('f1', ascending=False).head()
d121_rih_on_nih_f1.sort_values('f1', ascending=False).head()

mbv2_nih_on_rih_f1.sort_values('f1', ascending=False).head()
d121_nih_on_rih_f1.sort_values('f1', ascending=False).head()
mbv2_rih_on_rih_f1.sort_values('f1', ascending=False).head()
d121_rih_on_rih_f1.sort_values('f1', ascending=False).head()

def get_local_external_performance_diff(local, external):
    # Get optimal threshold based on local data 
    opt_thresh = local.sort_values('f1', ascending=False)['threshold'].iloc[0]
    opt_f1 = external.sort_values('f1', ascending=False)['f1'].iloc[0] 
    # Get F1 on external data using that threshold 
    ext_f1 = external[external.threshold == opt_thresh].f1.iloc[0] 
    pct_diff = (opt_f1 - ext_f1) / opt_f1 ; pct_diff *= 100. 
    print ('At threshold {thresh:.2f} performance decreased by {perf_diff:.1f}% from {localf1:.3f} to {externalf1:.3f}'.format(thresh=opt_thresh, perf_diff=pct_diff, localf1=opt_f1, externalf1=ext_f1))

get_local_external_performance_diff(mbv2_nih_on_nih_f1, mbv2_nih_on_rih_f1)
get_local_external_performance_diff(d121_nih_on_nih_f1, d121_nih_on_rih_f1)
get_local_external_performance_diff(mbv2_rih_on_rih_f1, mbv2_rih_on_nih_f1)
get_local_external_performance_diff(d121_rih_on_rih_f1, d121_rih_on_nih_f1)

nih_models_f1 = pd.concat([d121_nih_on_nih_f1, mbv2_nih_on_nih_f1, d121_nih_on_rih_f1, mbv2_nih_on_rih_f1])
rih_models_f1 = pd.concat([d121_rih_on_nih_f1, mbv2_rih_on_nih_f1, d121_rih_on_rih_f1, mbv2_rih_on_rih_f1])

# Make plots
import matplotlib as mpl, matplotlib.pyplot as plt
import seaborn as sns

sns.set() 
mpl.rc('font', family='DejaVu Sans')
fig, ax = plt.subplots(figsize=(12,8))
myplot = sns.lineplot(x='threshold', y='f1', data=nih_models_f1, style='model', hue='model', markers=True, size=50)
legend = ax.legend() 
fig.suptitle('F1 Scores at Various Thresholds\nNIH-Trained Models', fontsize=20)
ax.set(xlabel='Threshold', ylabel='F1 Score')
# Pad axis label space
ax.xaxis.labelpad = 10 ; ax.yaxis.labelpad = 15
# Change axis label font sizes
ax.xaxis.label.set_fontsize(14) 
ax.yaxis.label.set_fontsize(14) 
# Change tick labels 
ax.xaxis.set_ticks(np.linspace(0.05, 0.95, 10))
legend.texts[0].set_text('')
handles, labels = ax.get_legend_handles_labels()
labels[1:] = ['NIH-Dense  | NIH-CXR', 'NIH-Mobile | NIH-CXR', 'NIH-Dense  | RIH-CXR', 'NIH-Mobile | RIH-CXR']
ax.legend(handles=handles[1:], labels=labels[1:], prop={'family': 'monospace'})
plt.savefig('/users/ipan/Downloads/Figure1_NIH.svg', format='svg', dpi=1200)

sns.set() 
mpl.rc('font', family='DejaVu Sans')
fig, ax = plt.subplots(figsize=(12,8))
myplot = sns.lineplot(x='threshold', y='f1', data=rih_models_f1, style='model', hue='model', markers=True, size=50)
legend = ax.legend() 
fig.suptitle('F1 Scores at Various Thresholds\nRIH-Trained Models', fontsize=20)
ax.set(xlabel='Threshold', ylabel='F1 Score')
# Pad axis label space
ax.xaxis.labelpad = 10 ; ax.yaxis.labelpad = 20
# Change axis label font sizes
ax.xaxis.label.set_fontsize(14) 
ax.yaxis.label.set_fontsize(14) 
# Change tick labels 
#ax.yaxis.set_ticks(np.linspace(0.35, 0.85, 11))
ax.xaxis.set_ticks(np.linspace(0.05, 0.95, 10))
legend.texts[0].set_text('')
handles, labels = ax.get_legend_handles_labels()
labels[1:] = ['RIH-Dense  | NIH-CXR', 'RIH-Mobile | NIH-CXR', 'RIH-Dense  | RIH-CXR', 'RIH-Mobile | RIH-CXR']
ax.legend(handles=handles[1:], labels=labels[1:], loc='lower left', prop={'family': 'monospace'})
plt.savefig('/users/ipan/Downloads/Figure1_RIH.svg', format='svg', dpi=1200)

