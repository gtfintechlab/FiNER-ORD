import pandas as pd

# Load the CSV file into a DataFrame
df_test = pd.read_csv('./test/test.csv')
df_train = pd.read_csv('./train/train.csv')
df_val = pd.read_csv('./train/val.csv')

df = pd.concat([df_train, df_val, df_test], ignore_index=True)

# Group by 'gold_label' and count unique 'gold_tokens' for each group

grouped_df = df.groupby('gold_label')
unique_counts = grouped_df['gold_token'].nunique()
label_count = grouped_df.size()

# Print the result
print(label_count)
print(unique_counts)
