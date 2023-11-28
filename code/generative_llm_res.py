import numpy as np
import pandas as pd
import os
from nltk import word_tokenize

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from seqeval.metrics import classification_report as seqeval_clf_rpt

def decode(label_word):
    try:
        label_word = label_word.lower()
        if "other" in label_word:
            return 0
        elif "person_b" in label_word:
            return 1
        elif "person_i" in label_word:
            return 2
        elif "location_b" in label_word:
            return 3
        elif "location_i" in label_word:
            return 4
        elif "organisation_b" in label_word:
            return 5
        elif "organisation_i" in label_word:
            return 6
        else: 
            return -1
    except: 
        return -1


acc_list = []
f1_list = []
missing_perc_list = [] 

files = os.listdir('../data/llm_prompt_outputs')

files_xls = [f for f in files if 'gpt4' in f and '_10_' in f] # chatgpt or gpt4

for file in files_xls:
    df = pd.read_pickle('../data/llm_prompt_outputs/' + file)
    print(f"****{file}****")
    # print(df.head())

    true_labels = []
    predicted_labels = []

    y_true_entity_level_eval = []
    y_pred_entity_level_eval = []
    # mapping_dict = {
    #     'O': 'O',
    #     'PER_B': 'B-PER',
    #     'LOC_B': 'B-LOC',
    #     'PER_I': 'I-PER',
    #     'LOC_I': 'I-LOC',
    #     'ORG_B': 'B-ORG',
    #     'ORG_I': 'I-ORG'
    # }
    mapping_dict = {
        -1: 'B-MISSING',
        0: 'O',
        1: 'B-PER',
        3: 'B-LOC',
        2: 'I-PER',
        4: 'I-LOC',
        5: 'B-ORG',
        6: 'I-ORG'
    }
    # mapping_dict = {0: 'B-LOC', 1: 'I-LOC', 2: 'O', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-PER', 6: 'I-PER'}
    
    for cntr, index in enumerate(range(df.shape[0])):

        curr_true_list = []
        curr_pred_list = []

        true_labels_temp = df.loc[[index],['true_label']].values[0, 0]
        original_sent_temp = df.loc[[index],['original_sent']].values[0, 0]
        text_output_temp = df.loc[[index],['text_output']].values[0, 0]

        true_labels = true_labels + true_labels_temp

        if text_output_temp == None:
            for i in range(len(true_labels_temp)):
                predicted_labels.append(-1)
        else:
            original_sent_split = original_sent_temp.split("\n")
            text_output_split = text_output_temp.split("\n")
            
            sub_index_gold = 0
            sub_index_output = 0
            while sub_index_gold < len(original_sent_split): # or sub_index_output < len(text_output_split):
                try: 
                    text_output_token_label = text_output_split[sub_index_output].split(":")
                    # print(original_sent_split[sub_index_gold], text_output_token_label[0], text_output_token_label[1], "AAA")
                    if original_sent_split[sub_index_gold] == text_output_token_label[0]:
                        sub_index_output = sub_index_output + 1
                        predicted_labels.append(decode(text_output_token_label[1]))
                    elif original_sent_split[sub_index_gold] in text_output_token_label[0]:
                        if text_output_token_label[0].endswith(original_sent_split[sub_index_gold]):
                            sub_index_output = sub_index_output + 1  
                        predicted_labels.append(decode(text_output_token_label[1]))
                    else:
                        predicted_labels.append(-1)
                except:
                    predicted_labels.append(-1)
                sub_index_gold = sub_index_gold + 1

                #### find examples of LOC and ORG confusion ####
                # true_idx = len(predicted_labels) - 1
                # pred_idx = -1
                # if (true_labels[true_idx] in [3, 4] and predicted_labels[pred_idx] in [5, 6]) or (true_labels[true_idx] in [5, 6] and predicted_labels[pred_idx] in [3, 4]):
                #     print("true:", true_labels[true_idx])
                #     print("predicted:", predicted_labels[pred_idx])
                #     print(sub_index_gold)
                #     print(original_sent_split[sub_index_gold-1])
                #     # if sub_index_gold > 0 and sub_index_gold < len(original_sent_split):
                #     #     print(original_sent_split[sub_index_gold-2])
                #     print(original_sent_split)
                #     print()

                #### seqeval ####
                true_idx = len(predicted_labels) - 1
                pred_idx = -1
                curr_true_list.append(true_labels[true_idx])
                curr_pred_list.append(predicted_labels[pred_idx])
                # if (true_labels[true_idx] in [3, 4] and predicted_labels[pred_idx] in [5, 6]) or (true_labels[true_idx] in [5, 6] and predicted_labels[pred_idx] in [3, 4]):
                #     print("true:", true_labels[true_idx])
                #     print("predicted:", predicted_labels[pred_idx])
                #     print(sub_index_gold)
                #     print(original_sent_split[sub_index_gold-1])
                #     # if sub_index_gold > 0 and sub_index_gold < len(original_sent_split):
                #     #     print(original_sent_split[sub_index_gold-2])
                #     print(original_sent_split)
                #     print()

            mapped_true_labels = [mapping_dict[item] for item in curr_true_list]
            mapped_pred_labels = [mapping_dict[item] for item in curr_pred_list]

            y_true_entity_level_eval.append(mapped_true_labels)
            y_pred_entity_level_eval.append(mapped_pred_labels)

            # print('curr_true')
            # print(curr_true_list)
            # print()
            # print("curr_pred")
            # print(curr_pred_list)
            # print()
            # print('mapped true')
            # print(mapped_true_labels)
            # print()
            # print('mapped pred')
            # print(mapped_pred_labels)
            # print()
            # print('true eval')
            # print(y_true_entity_level_eval)
            # print()
            # print('pred eval')
            # print(y_pred_entity_level_eval)
        
    # print(true_labels)
    # print(predicted_labels)
        # if cntr > 5:
        #     break

    acc_list.append(accuracy_score(true_labels, predicted_labels))
    f1_list.append(f1_score(true_labels, predicted_labels, average='weighted'))
    missing_perc_list.append((predicted_labels.count(-1)/len(predicted_labels))*100.0)
    # print(classification_report(predicted_labels, true_labels, digits=7))
    print(seqeval_clf_rpt(y_true_entity_level_eval, y_pred_entity_level_eval, digits=4))

print("f1 score mean: ", format(np.mean(f1_list), '.4f'))
print("f1 score std: ", format(np.std(f1_list), '.4f'))
print("Percentage of cases when didn't follow instruction: ", format(np.mean(missing_perc_list), '.4f'), "\n")
