import pandas as pd

df = pd.read_excel("test_results/Subject_1_028.xlsx", engine='openpyxl', sheet_name=None)
questionnaire = df.pop('Questionnaire')

a = df[0].iloc[:, [0, 1]]

room = a.columns[0]

a = a.rename(columns={a.columns[1]: f'{room}_answer'})

b = a.iloc[0:2+2, :]

condition = b.iloc[0, 0]
df_2 = pd.DataFrame(columns=['Condition', 'Answer', 'Speaker'])

if condition == 'NONE':
    if b.iloc[1, 1] == 'Yes':
        df_2.loc[len(df_2.index)] = [condition, 1, 'None']
    else:
        speaker = b.iloc[2:, ].dropna().iloc[0, 0]
        df_2.loc[len(df_2.index)] = [condition, 0, speaker]

else:
    row = b.iloc[2:, ].dropna()
    df_2.loc[len(df_2.index)] = [condition, row.iloc[0, 1], row.iloc[0, 0]]


