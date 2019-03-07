'''
This script assigns training/validation/test splits to the NIH CXR14
dataset. Training: 70%, validation: 10%, test: 20%.

There are 3 different train/val splits and 1 test set. 

Note that we exclude images with no findings if that patient has had 
other images with findings. By not doing so, performance is low for 
findings vs. no finding classification. 

No patient overlap across train/val/test splits. A patient can have 
one to many images, so we try to ensure that the train/val/test 
distribution is maintained at both patient- and image-levels. 

We attempt to stratify the splits based on finding. Because an image
can have multiple findings, this script is more complex than one might 
expect. 
'''

import argparse 

parser = argparse.ArgumentParser() 

parser.add_argument('nih_csv', type=str) 
parser.add_argument('cv_csv', type=str) 
parser.add_argument('--seed', type=int, nargs='?', const=88, default=88)

args = parser.parse_args()  


#######################
# PREPARE NIH DATASET # 
#######################

from tqdm import tqdm 

import pandas as pd 
import numpy as np 

def convert_text_labels_to_matrix(df): 
    findings = ["Hernia", "Pneumonia", "Fibrosis", "Edema", "Emphysema", 
                "Cardiomegaly", "Pleural_Thickening", "Consolidation", 
                "Pneumothorax", "Mass", "Nodule", "Atelectasis", "Effusion",
                "Infiltration", "No Finding"]
    for find in findings: 
        df[find] = 0 
    for rownum, row in tqdm(df.iterrows(), total=len(df)): 
        labels = row['Finding Labels'].split('|')
        for lab in labels: 
            df.loc[rownum, lab] = 1
    return df 

cxr = pd.read_csv(args.nih_csv)  
cxr = convert_text_labels_to_matrix(cxr) 

##################################
# ===== ASSIGN DATA SPLITS ===== #
##################################

def are_all_xrays_negative(pid, df):
    tmp_df = df[df["Patient ID"] == pid] 
    num_no_findings = np.sum(tmp_df["No Finding"])
    if num_no_findings == len(tmp_df): 
        return True 
    else:
        return False 

pid_no_findings = [pid for pid in np.unique(cxr["Patient ID"]) if are_all_xrays_negative(pid, cxr)]
cxr_no_findings = cxr[(cxr["Patient ID"].isin(pid_no_findings))]

pid_findings = list(set(cxr["Patient ID"]) - set(pid_no_findings))
cxr_findings = cxr[(cxr["Patient ID"].isin(pid_findings))]
cxr_findings = cxr_findings[cxr_findings["No Finding"] == 0]

#Hernia              227
#Pneumonia          1353
#Fibrosis           1686
#Edema              2303
#Emphysema          2516
#Cardiomegaly       2772
#Pleural_Thickening 3385
#Consolidation      4667
#Pneumothorax       5298 
#Mass               5746
#Nodule             6323
#Atelectasis       11535
#Effusion          13307
#Infiltration      19870

def get_splits_for_each_finding(cxr_findings, finding, already_assigned=[], num_val=3):
    cxr_findings = cxr_findings[cxr_findings[finding] == 1]
    cxr_findings = cxr_findings[~(cxr_findings["Patient ID"].isin(already_assigned))]
    pid_pos_image_counts = np.unique(cxr_findings["Patient ID"], return_counts=True)
    pid_pos_image_counts_df = pd.DataFrame({"pid": pid_pos_image_counts[0],
                                            "num": pid_pos_image_counts[1]})
    pid_pos_image_counts_df = pid_pos_image_counts_df.sort_values("num", ascending=False)
    pid_pos_top50 = pid_pos_image_counts_df["pid"].iloc[:(len(pid_pos_image_counts_df) / 2)] 
    pid_pos_bot50 = pid_pos_image_counts_df["pid"].iloc[(len(pid_pos_image_counts_df) / 2):] 
    # Assign test first 
    pid_pos_test_top50 = np.random.choice(pid_pos_top50, int(0.2*len(pid_pos_top50)), replace=False)
    pid_pos_test_bot50 = np.random.choice(pid_pos_bot50, int(0.2*len(pid_pos_bot50)), replace=False) 
    #
    pid_pos_not_test_top50 = list(set(pid_pos_top50) - set(pid_pos_test_top50))
    pid_pos_not_test_bot50 = list(set(pid_pos_bot50) - set(pid_pos_test_bot50))
    already_assigned_to_valid = [] 
    pid_pos_train_top50_list = [] 
    pid_pos_valid_top50_list = []
    for each_val in range(num_val): 
        pid_pos_not_test_or_valid_top50 = list(set(pid_pos_not_test_top50) - set(already_assigned_to_valid))
        # Sample validation split first
        pid_pos_valid_top50 = np.random.choice(pid_pos_not_test_or_valid_top50, int(0.1*len(pid_pos_top50)), replace=False) 
        pid_pos_train_top50 = list(set(pid_pos_not_test_top50) - set(pid_pos_valid_top50))
        already_assigned_to_valid.extend(pid_pos_valid_top50) 
        pid_pos_train_top50_list.append(pid_pos_train_top50) 
        pid_pos_valid_top50_list.append(pid_pos_valid_top50) 
    already_assigned_to_valid = [] 
    pid_pos_train_bot50_list = [] 
    pid_pos_valid_bot50_list = []
    for each_val in range(num_val): 
        pid_pos_not_test_or_valid_bot50 = list(set(pid_pos_not_test_bot50) - set(already_assigned_to_valid))
        # Sample validation split first
        pid_pos_valid_bot50 = np.random.choice(pid_pos_not_test_or_valid_bot50, int(0.1*len(pid_pos_bot50)), replace=False) 
        pid_pos_train_bot50 = list(set(pid_pos_not_test_bot50) - set(pid_pos_valid_bot50))
        already_assigned_to_valid.extend(pid_pos_valid_bot50) 
        pid_pos_train_bot50_list.append(pid_pos_train_bot50) 
        pid_pos_valid_bot50_list.append(pid_pos_valid_bot50) 
    pid_pos_train_list = []
    pid_pos_valid_list = [] 
    for each_val in range(num_val): 
        pid_pos_train_list.append(np.concatenate((pid_pos_train_top50_list[each_val], pid_pos_train_bot50_list[each_val])))
        pid_pos_valid_list.append(np.concatenate((pid_pos_valid_top50_list[each_val], pid_pos_valid_bot50_list[each_val])))
    pid_pos_test = np.concatenate((pid_pos_test_top50, pid_pos_test_bot50))
    return pid_pos_train_list, pid_pos_valid_list, pid_pos_test 

def turn_list_of_arrays_into_list(list_of_arrays):
    single_list = []
    for _ in list_of_arrays:
        single_list.extend(list(_))
    return single_list

def assign_splits(seed=88, num_val=3):
    np.random.seed(seed)
    # NEGATIVES
    pid_neg_image_counts = np.unique(cxr_no_findings["Patient ID"], return_counts=True)
    pid_neg_image_counts_df = pd.DataFrame({"pid": pid_neg_image_counts[0],
                                            "num": pid_neg_image_counts[1]})
    pid_neg_image_counts_df = pid_neg_image_counts_df.sort_values("num", ascending=False)
    pid_neg_top50 = pid_neg_image_counts_df["pid"].iloc[:(len(pid_neg_image_counts_df) / 2)] 
    pid_neg_bot50 = pid_neg_image_counts_df["pid"].iloc[(len(pid_neg_image_counts_df) / 2):] 
    # Assign test first 
    pid_neg_test_top50 = np.random.choice(pid_neg_top50, int(0.2*len(pid_neg_top50)), replace=False)
    pid_neg_test_bot50 = np.random.choice(pid_neg_bot50, int(0.2*len(pid_neg_bot50)), replace=False) 
    #
    pid_neg_not_test_top50 = list(set(pid_neg_top50) - set(pid_neg_test_top50))
    pid_neg_not_test_bot50 = list(set(pid_neg_bot50) - set(pid_neg_test_bot50))
    already_assigned_to_valid = [] 
    pid_neg_train_top50_list = [] 
    pid_neg_valid_top50_list = []
    for each_val in range(num_val): 
        pid_neg_not_test_or_valid_top50 = list(set(pid_neg_not_test_top50) - set(already_assigned_to_valid))
        # Sample validation split first
        pid_neg_valid_top50 = np.random.choice(pid_neg_not_test_or_valid_top50, int(0.1*len(pid_neg_top50)), replace=False) 
        pid_neg_train_top50 = list(set(pid_neg_not_test_top50) - set(pid_neg_valid_top50))
        already_assigned_to_valid.extend(pid_neg_valid_top50) 
        pid_neg_train_top50_list.append(pid_neg_train_top50) 
        pid_neg_valid_top50_list.append(pid_neg_valid_top50) 
    already_assigned_to_valid = [] 
    pid_neg_train_bot50_list = [] 
    pid_neg_valid_bot50_list = []
    for each_val in range(num_val): 
        pid_neg_not_test_or_valid_bot50 = list(set(pid_neg_not_test_bot50) - set(already_assigned_to_valid))
        # Sample validation split first
        pid_neg_valid_bot50 = np.random.choice(pid_neg_not_test_or_valid_bot50, int(0.1*len(pid_neg_bot50)), replace=False) 
        pid_neg_train_bot50 = list(set(pid_neg_not_test_bot50) - set(pid_neg_valid_bot50))
        already_assigned_to_valid.extend(pid_neg_valid_bot50) 
        pid_neg_train_bot50_list.append(pid_neg_train_bot50) 
        pid_neg_valid_bot50_list.append(pid_neg_valid_bot50) 
    pid_neg_train_list = []
    pid_neg_valid_list = [] 
    for each_val in range(num_val): 
        pid_neg_train_list.append(np.concatenate((pid_neg_train_top50_list[each_val], pid_neg_train_bot50_list[each_val])))
        pid_neg_valid_list.append(np.concatenate((pid_neg_valid_top50_list[each_val], pid_neg_valid_bot50_list[each_val])))
    pid_neg_test = np.concatenate((pid_neg_test_top50, pid_neg_test_bot50))
    # POSITIVES 
    already_assigned = [] 
    # Start with least common finding --> most common 
    hernia_train, hernia_valid, hernia_test = get_splits_for_each_finding(cxr_findings, "Hernia", [], num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(hernia_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(hernia_valid)) ; already_assigned.extend(hernia_test)
    pneumo_train, pneumo_valid, pneumo_test = get_splits_for_each_finding(cxr_findings, "Pneumonia", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(pneumo_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(pneumo_valid)) ; already_assigned.extend(pneumo_test)
    fibros_train, fibros_valid, fibros_test = get_splits_for_each_finding(cxr_findings, "Fibrosis", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(fibros_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(fibros_valid)) ; already_assigned.extend(fibros_test)
    edema_train, edema_valid, edema_test = get_splits_for_each_finding(cxr_findings, "Edema", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(edema_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(edema_valid)) ; already_assigned.extend(edema_test)
    emphys_train, emphys_valid, emphys_test = get_splits_for_each_finding(cxr_findings, "Emphysema", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(emphys_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(emphys_valid)) ; already_assigned.extend(emphys_test)
    cardio_train, cardio_valid, cardio_test = get_splits_for_each_finding(cxr_findings, "Cardiomegaly", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(cardio_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(cardio_valid)) ; already_assigned.extend(cardio_test)
    plthck_train, plthck_valid, plthck_test = get_splits_for_each_finding(cxr_findings, "Pleural_Thickening", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(plthck_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(plthck_valid)) ; already_assigned.extend(plthck_test)
    consld_train, consld_valid, consld_test = get_splits_for_each_finding(cxr_findings, "Consolidation", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(consld_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(consld_valid)) ; already_assigned.extend(consld_test)
    pnmthx_train, pnmthx_valid, pnmthx_test = get_splits_for_each_finding(cxr_findings, "Pneumothorax", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(pnmthx_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(pnmthx_valid)) ; already_assigned.extend(pnmthx_test)
    mass_train, mass_valid, mass_test = get_splits_for_each_finding(cxr_findings, "Mass", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(mass_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(mass_valid)) ; already_assigned.extend(mass_test)
    nodule_train, nodule_valid, nodule_test = get_splits_for_each_finding(cxr_findings, "Nodule", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(nodule_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(nodule_valid)) ; already_assigned.extend(nodule_test)
    atelec_train, atelec_valid, atelec_test = get_splits_for_each_finding(cxr_findings, "Atelectasis", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(atelec_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(atelec_valid)) ; already_assigned.extend(atelec_test)
    effuse_train, effuse_valid, effuse_test = get_splits_for_each_finding(cxr_findings, "Effusion", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(effuse_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(effuse_valid)) ; already_assigned.extend(effuse_test)
    infilt_train, infilt_valid, infilt_test = get_splits_for_each_finding(cxr_findings, "Infiltration", already_assigned, num_val) 
    already_assigned.extend(turn_list_of_arrays_into_list(infilt_train)) ; already_assigned.extend(turn_list_of_arrays_into_list(infilt_valid)) ; already_assigned.extend(infilt_test)
    pid_pos_train_list = [] 
    pid_pos_valid_list = [] 
    for each_val in range(num_val):
        pid_pos_train = np.concatenate((hernia_train[each_val], pneumo_train[each_val], fibros_train[each_val],
                                        edema_train[each_val], emphys_train[each_val], cardio_train[each_val],
                                        plthck_train[each_val], consld_train[each_val], pnmthx_train[each_val], 
                                        mass_train[each_val], nodule_train[each_val], atelec_train[each_val],
                                        effuse_train[each_val], infilt_train[each_val]))
        pid_pos_valid = np.concatenate((hernia_valid[each_val], pneumo_valid[each_val], fibros_valid[each_val],
                                        edema_valid[each_val], emphys_valid[each_val], cardio_valid[each_val],
                                        plthck_valid[each_val], consld_valid[each_val], pnmthx_valid[each_val], 
                                        mass_valid[each_val], nodule_valid[each_val], atelec_valid[each_val],
                                        effuse_valid[each_val], infilt_valid[each_val]))
        pid_pos_train_list.append(pid_pos_train)
        pid_pos_valid_list.append(pid_pos_valid)
    pid_pos_test  = np.concatenate((hernia_test, pneumo_test, fibros_test,
                                    edema_test, emphys_test, cardio_test,
                                    plthck_test, consld_test, pnmthx_test, 
                                    mass_test, nodule_test, atelec_test,
                                    effuse_test, infilt_test))
    # Make a DataFrame with IDs, labels, and splits 
    pid_neg_train_image_indices = [] 
    for each_neg_train in pid_neg_train_list:
        pid_neg_train_image_indices.append(cxr_no_findings[cxr_no_findings["Patient ID"].isin(each_neg_train)]["Image Index"])
    pid_pos_train_image_indices = [] 
    for each_pos_train in pid_pos_train_list:
        pid_pos_train_image_indices.append(cxr_findings[cxr_findings["Patient ID"].isin(each_pos_train)]["Image Index"])        
    pid_neg_valid_image_indices = [] 
    for each_neg_valid in pid_neg_valid_list:
        pid_neg_valid_image_indices.append(cxr_no_findings[cxr_no_findings["Patient ID"].isin(each_neg_valid)]["Image Index"])
    pid_pos_valid_image_indices = [] 
    for each_pos_valid in pid_pos_valid_list:
        pid_pos_valid_image_indices.append(cxr_findings[cxr_findings["Patient ID"].isin(each_pos_valid)]["Image Index"])        
    pid_neg_test = cxr_no_findings[cxr_no_findings["Patient ID"].isin(pid_neg_test)]["Image Index"]
    pid_pos_test = cxr_findings[cxr_findings["Patient ID"].isin(pid_pos_test)]["Image Index"]
    df_list = []
    for each_val in range(num_val): 
        tmp_train_df = pd.DataFrame({"pid": np.concatenate((pid_neg_train_image_indices[each_val], pid_pos_train_image_indices[each_val])),
                                    "y_true": np.concatenate((np.repeat(0, len(pid_neg_train_image_indices[each_val])), np.repeat(1, len(pid_pos_train_image_indices[each_val]))))})
        tmp_train_df["split{}".format(each_val)] = np.repeat("train", len(tmp_train_df))
        tmp_valid_df = pd.DataFrame({"pid": np.concatenate((pid_neg_valid_image_indices[each_val], pid_pos_valid_image_indices[each_val])),
                                    "y_true": np.concatenate((np.repeat(0, len(pid_neg_valid_image_indices[each_val])), np.repeat(1, len(pid_pos_valid_image_indices[each_val]))))})
        tmp_valid_df["split{}".format(each_val)] = np.repeat("valid", len(tmp_valid_df))
        tmp_test_df = pd.DataFrame({"pid": np.concatenate((pid_neg_test, pid_pos_test)),
                                    "y_true": np.concatenate((np.repeat(0, len(pid_neg_test)), np.repeat(1, len(pid_pos_test))))})
        tmp_test_df["split{}".format(each_val)] = np.repeat("test", len(tmp_test_df))
        tmp_df = tmp_train_df.append(tmp_valid_df)
        tmp_df = tmp_df.append(tmp_test_df)
        tmp_df = tmp_df.sort_values("pid")
        df_list.append(tmp_df) 
    final_df = df_list[0] 
    for each_val in range(1, num_val): 
        final_df["split{}".format(each_val)] = list(df_list[each_val]["split{}".format(each_val)])
    return final_df 

cxr_df_splits = assign_splits(seed=args.seed)
cxr_df_splits = cxr_df_splits.merge(cxr[["Image Index", "Hernia", "Pneumonia", "Fibrosis", "Edema", "Emphysema", "Cardiomegaly", "Pleural_Thickening", "Consolidation", "Pneumothorax", "Mass", "Nodule", "Atelectasis", "Effusion", "Infiltration", "No Finding"]], left_on="pid", right_on="Image Index")
cxr_df_splits.to_csv(args.cv_csv, index=False)


