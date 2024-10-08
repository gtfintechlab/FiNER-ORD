import pandas as pd
import os


df = pd.read_csv('../data/grid_search_results/roberta-large_2_0/results.csv')
df_temp = df.groupby(['Learning Rate', 'Batch Size'], as_index=False).agg(
{
    "Val F1": ["mean"],
    "Test F1": ["mean", "std"], 
    "Fine Tuning Time(m)": ["mean"],
    "Test Labeling Time(m)": ["mean"]
}
)
df_temp.columns = ['Learning Rate', 'Batch Size', 'mean Val F1 Score', 'mean Test F1 Score', 'std Test F1 Score', 'mean Fine Tuning Time(m)', 'mean Test Labeling Time(m)']

max_element = df_temp.iloc[df_temp['mean Val F1 Score'].idxmax()] 
print(max_element)
