# 
# Based on the master list in jester-v1-train - create a subset based on num_samples and selected_classes
# 
# Edmund Leow
num_samples = -1 # use -1 to use all possible samples, otherwise specify total samples (training, validation, test) for each class
seed = 123
selected_classes = ['Doing other things', 'No gesture', 'Stop Sign']  # basic gestures
selected_classes += ['Swiping Down', 'Swiping Left', 'Swiping Right', 'Swiping Up']  # swiping
selected_classes += ['Turning Hand Clockwise', 'Turning Hand Counterclockwise']  # turning

csv_path = './annotations/hand_gestures_labels.csv'  # file containing master set of all samples and labels
label_out_path = './annotations/mew_jester-v1-labels.csv'
data_split = [
    ('./annotations/mew_all_jester-v1-train.csv', 0.8), 
    ('./annotations/mew_all_jester-v1-validation.csv', 0.1), 
    ('./annotations/mew_all_jester-v1-test.csv', 0.1)
]


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

gest = pd.read_csv(csv_path, dtype={'number': np.int32})  # all samples, all classes

filtered = gest.loc[gest['label'].isin(selected_classes)] # all samples for selected classes
# filtered = filtered.sample(frac=1).reset_index(drop=True)

df_samples= pd.DataFrame({'number': [], 'label': []})
normalised_split = [d[1] for d in data_split]
normalised_split = [n/sum(normalised_split) for n in normalised_split]

if num_samples > 0:
    for c in selected_classes:
        f = filtered.loc[filtered['label'] == c] 
        f = f.sample(n=num_samples, random_state=seed)
        # df_train = f.sample(frac=normalised_split[0])
        df_samples = df_samples.append(f)
else:
    df_samples = filtered

# Split dataset to train, validation, and test
test, train = train_test_split(df_samples, test_size=normalised_split[0], random_state=seed, stratify=df_samples.label)
test, val = train_test_split(test, test_size=normalised_split[1]/sum(normalised_split[1:]), random_state=seed, stratify=test.label)
print('Train', 'Validation', 'Test')
print(len(train), len(val), len(test))

# Save to csv files (with ; as separator to be inline with Jester file format)
train.astype({'number': int}).to_csv(data_split[0][0], index=None, header=False, sep=';')
val.astype({'number': int}).to_csv(data_split[1][0], index=None, header=False, sep=';')
test.astype({'number': int}).to_csv(data_split[2][0], index=None, header=False, sep=';')
pd.DataFrame(selected_classes).to_csv(label_out_path, index=None, header=False)

print()
