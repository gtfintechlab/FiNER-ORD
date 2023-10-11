"""
Docs:
1. https://developers.generativeai.google/tutorials/chat_quickstart
2. https://developers.generativeai.google/tutorials/text_quickstart
3. https://developers.generativeai.google/guide/palm_api_overview
"""

import os,sys
import google.generativeai as palm
import pandas as pd
from time import sleep, time
from datetime import date
today = date.today()


sys.path.insert(0, '/home/research/git repos/zero-shot-finance')
from api_keys import APIKeyConstants
palm.configure(api_key=APIKeyConstants.PALM_API_KEY)

models = [m for m in palm.list_models() if 'generateText' in m.supported_generation_methods]
model = models[0].name
model_name = model.replace("/", "_")



start_t = time()
# load training data
test_data_path = "../data/test/test.csv"
data_df = pd.read_csv(test_data_path)

grouped_df = data_df.groupby(['doc_idx', 'sent_idx']).agg({'gold_label':lambda x: list(x), 'gold_token':lambda x: list(x)}).reset_index()
grouped_df.columns = ['doc_idx', 'sent_idx', 'gold_label', 'gold_token']



output_list = []
for index in range(grouped_df.shape[0]):
    token_list = grouped_df.loc[[index],['gold_token']].values[0, 0]
    label_list = grouped_df.loc[[index],['gold_label']].values[0, 0]
    sen = '\n'.join(token_list)

    message = "Discard all the previous instructions. Behave like you are an expert named entity identifier. Below a sentence is tokenized and each line contains a word token from the sentence. Identify 'Person', 'Location', and 'Organisation' from them and label them. If the entity is multi token use post-fix _B for the first label and _I for the remaining token labels for that particular entity. The start of the separate entity should always use _B post-fix for the label. If the token doesn't fit in any of those three categories or is not a named entity label it 'Other'. Do not combine words yourself. Use a colon to separate token and label. So the format should be token:label. \n\n" + sen

    prompt_json = [
            {"role": "user", "content": message},
    ]
    try:
        # response = palm.chat(messages=message)
        completion = palm.generate_text(
                            model=model,
                            prompt=message,
                            temperature=0.0,
                            # The maximum length of the response
                            max_output_tokens=1000,
                        )
    except Exception as e:
        print(e)
        i = i - 1
        sleep(10.0)

    answer = completion.result#response.last 
    
    output_list.append([label_list, sen, answer])
    sleep(2.0) 



results = pd.DataFrame(output_list, columns=["true_label", "original_sent", "text_output"])

time_taken = int((time() - start_t)/60.0)

results.to_pickle(f'../data/llm_prompt_outputs/palm_{today.strftime("%d_%m_%Y")}_{time_taken}')

