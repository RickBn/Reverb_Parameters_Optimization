import pandas as pd

df = pd.read_excel("test_results/Subject_1_028.xlsx", engine='openpyxl', sheet_name=None)
questionnaire = df.pop('Questionnaire')

rooms = ['LIVING_ROOM', 'METU', '3D_MARCo']
conditions = ['NONE', 'HOA_Bin', 'FV']

df_2 = pd.DataFrame(columns=['Complexity', 'Room', 'Condition', 'Answer', 'Speaker'])

for complexity in df.keys():
    for r in range(0, len(rooms)):
        comp_df = df[complexity].iloc[:, [r * 2, (r * 2) + 1]]

        room = comp_df.columns[0]

        comp_df = comp_df.rename(columns={comp_df.columns[1]: f'{room}_answer'})

        n_comp = int(complexity.split('_')[1]) + 2

        for i in range(0, len(conditions)):
            b = comp_df.iloc[i * n_comp:(i * n_comp) + n_comp, :]

            condition = b.iloc[0, 0]

            if condition == 'NONE':
                if b.iloc[1, 1] == 'Yes':
                    df_2.loc[len(df_2.index)] = [complexity, room, condition, 1, 'None']
                else:
                    speaker = b.iloc[2:, ].dropna().iloc[0, 0]
                    df_2.loc[len(df_2.index)] = [complexity, room, condition, 0, speaker]

            else:
                if b.iloc[1, 1] == 'No':
                    row = b.iloc[2:, ].dropna()
                    df_2.loc[len(df_2.index)] = [complexity, room, condition, row.iloc[0, 1], row.iloc[0, 0]]
                else:
                    df_2.loc[len(df_2.index)] = [complexity, room, condition, 0, 'None']
