import sys
from scripts.utils.directory_functions import directory_filter

import pandas as pd

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        test_path = 'test_results/'
    else:
        test_path = sys.argv[1]

    df = pd.DataFrame(columns=['subject_matlab_id', 'subject_progressive_id', 'hrtf',
                               'complexity', 'room', 'condition',
                               'same', 'impostor',
                               'speaker', 'position'])
    df.index.name = 'trial_id'

    preliminary = pd.DataFrame(columns=['subject_matlab_id', 'subject_progressive_id', 'hrtf', 'room', 'condition', 'speaker', 'externalization'])
    preliminary.index.name = 'trial_id_per_subject'
    subject_info = pd.DataFrame(columns=['subject_matlab_id', 'subject_progressive_id', 'age', 'sex', 'reverberation', 'vr', 'impairment', 'duration'])

    for test_file in directory_filter(test_path):
        subject_progressive_id, subject_matlab_id, subject_hrtf = test_file.replace('.xlsx', "").split('_')[0:3]
        results = pd.read_excel(f'{test_path}{test_file}', engine='openpyxl', sheet_name=None)# , error_bad_lines=False)
        questionnaire = results.pop('Questionnaire')

        questionnaire.columns = map(str.lower, questionnaire.columns)
        questionnaire.columns = [s.replace(' ', '') for s in questionnaire.columns]
        pre_df = questionnaire[['room', 'condition', 'speaker', 'externalization']]
        pre_df.insert(0, 'hrtf', subject_hrtf)
        pre_df.insert(0, 'subject_progressive_id', subject_progressive_id)
        pre_df.insert(0, 'subject_matlab_id', subject_matlab_id)

        preliminary = preliminary.append(pre_df)

        ans = questionnaire['answer']
        subject_info.loc[len(subject_info.index)] = [subject_matlab_id, subject_progressive_id, ans[0], ans[1],
                                                     ans[2], ans[3], ans[4], ans[7]]
        
        rooms = ['LIVING_ROOM', 'METU', '3D_MARCo']
        conditions = ['NONE', 'HOA_Bin', 'FV']
        
        for complexity in results.keys():
            for r in range(0, len(rooms)):
                comp_df = results[complexity].iloc[:, [r * 2, (r * 2) + 1]]
        
                room = comp_df.columns[0]
        
                comp_df = comp_df.rename(columns={comp_df.columns[1]: f'{room}_answer'})
        
                n_comp = int(complexity.split('_')[1]) + 2
        
                for i in range(0, len(conditions)):
                    b = comp_df.iloc[i * n_comp:(i * n_comp) + n_comp, :]
        
                    condition = b.iloc[0, 0]
        
                    if condition == 'NONE':
                        if b.iloc[1, 1] == 'Yes':
                            df.loc[len(df.index)] = [subject_matlab_id, subject_progressive_id, subject_hrtf,
                                                     complexity, room, condition, 1, 1,
                                                     'None', 'None']
                        else:
                            position, speaker = b.iloc[2:, ].dropna().iloc[0, 0].split(' - ')

                            df.loc[len(df.index)] = [subject_matlab_id, subject_progressive_id, subject_hrtf,
                                                     complexity, room, condition, 0, 0,
                                                     speaker, position]
        
                    else:
                        if b.iloc[1, 1] == 'No':
                            row = b.iloc[2:, ].dropna()
                            position, speaker = row.iloc[0, 0].split(' - ')

                            df.loc[len(df.index)] = [subject_matlab_id, subject_progressive_id, subject_hrtf,
                                                     complexity, room, condition, 1,
                                                     row.iloc[0, 1],
                                                     speaker, position]
                        else:
                            df.loc[len(df.index)] = [subject_matlab_id, subject_progressive_id, subject_hrtf,
                                                     complexity, room, condition, 0, 0,
                                                     'None', 'None']

    df.to_csv('test_dataframes/experiment.csv')
    preliminary.to_csv('test_dataframes/preliminary.csv')
    subject_info.to_csv('test_dataframes/subject_info.csv', index=False)
