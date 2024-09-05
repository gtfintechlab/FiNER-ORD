import pandas as pd

df_1_train = pd.read_csv("../data/train/news_acl_42_1_1_gold_data_generated_data.csv.gz")
df_2_train = pd.read_csv("../data/train/conll_seed_42_train_split_1_1_gold_data_generated_data.csv.gz")

df_train = df_1_train.append(df_2_train, ignore_index=True)

df_train.to_csv("../data/train/combined_train_finer_conll.csv", index=False)

df_1_val = pd.read_csv("../data/train/news_acl_42_val_split_1_1_gold_data_generated_data.csv.gz")
df_2_val = pd.read_csv("../data/train/conll_seed_42_val_split_1_1_gold_data_generated_data.csv.gz")


df_val = df_1_val.append(df_2_val, ignore_index=True)

df_val.to_csv("../data/train/combined_val_finer_conll.csv", index=False)