import sys
from scripts.utils.directory_functions import directory_filter

import pandas as pd
from openpyxl import load_workbook

import string
abc = list(string.ascii_uppercase)

speakers_sex = {'ALEX': 'M',
                'DAVID': 'M',
                'MARIA': 'F',
                'SUSAN': 'F'}

def ilocnan(df, r):
    try:
        res = df[r]
    except (IndexError, KeyError):
        res = float('nan')

    return res


if __name__ == "__main__":
    if len(sys.argv) <= 1:
        test_path = 'test_results/'
    else:
        test_path = sys.argv[1]

    df = pd.DataFrame(columns=['subject_matlab_id', 'subject_progressive_id', 'hrtf',
                               'complexity', 'room', 'condition', 'same_room_answer', 'same_correct_answer',
                               'speaker_impostor', 'sex_impostor', 'position_impostor', 'speaker_answer', 'position_answer',
                               'impostor_correct_answer', 'speaker1', 'speaker2', 'speaker3', 'speaker4',
                               'speaker1_sex', 'speaker2_sex', 'speaker3_sex', 'speaker4_sex'])
    df.index.name = 'trial_id'

    preliminary = pd.DataFrame(columns=['subject_matlab_id', 'subject_progressive_id', 'hrtf', 'room', 'condition',
                                        'speaker', 'externalization'])

    subject_info = pd.DataFrame(columns=['subject_matlab_id', 'subject_progressive_id', 'age', 'sex', 'reverberation',
                                         'vr', 'impairment', 'duration', 'identification_strategy', 'confidence',
                                         'vr_sickness', 'other_comments'])

    for test_file in directory_filter(test_path):
        subject_progressive_id, subject_matlab_id, subject_hrtf = test_file.replace('.xlsx', "").split('_')[0:3]
        results = pd.read_excel(f'{test_path}{test_file}', engine='openpyxl', sheet_name=None)# , error_bad_lines=False)
        results_wb = load_workbook(f'{test_path}{test_file}')
        questionnaire = results.pop('Questionnaire')

        questionnaire.columns = map(str.lower, questionnaire.columns)
        questionnaire.columns = [s.replace(' ', '') for s in questionnaire.columns]
        pre_df = questionnaire[['room', 'condition', 'speaker', 'externalization']]
        pre_df.insert(0, 'hrtf', subject_hrtf)
        pre_df.insert(0, 'subject_progressive_id', subject_progressive_id)
        pre_df.insert(0, 'subject_matlab_id', subject_matlab_id)

        preliminary = preliminary.append(pre_df)

        ans = questionnaire['answer']
        feedback = questionnaire['feedback']
        subject_info.loc[len(subject_info.index)] = [subject_matlab_id, subject_progressive_id, ans[0], ans[1],
                                                     ans[2], ans[3], ans[4], ans[7], feedback[0], feedback[1],
                                                     feedback[2], feedback[3]]
        
        rooms = ['LIVING_ROOM', 'METU', '3D_MARCo']
        conditions = ['NONE', 'HOA_Bin', 'FV']
        
        for complexity in results.keys():
            complexity_n = int(complexity.split('_')[1])
            for r in range(0, len(rooms)):
                comp_df = results[complexity].iloc[:, [r * 2, (r * 2) + 1]]

                room = comp_df.columns[0].strip().upper().replace(' ', '_')
        
                comp_df = comp_df.rename(columns={comp_df.columns[1]: f'{room}_answer'})
        
                n_comp = complexity_n + 2
        
                for i in range(0, len(conditions)):
                    b = comp_df.iloc[i * n_comp:(i * n_comp) + n_comp, :]
                    b_wb = results_wb[complexity][
                        f'{abc[r * 2]}{i * n_comp+2}:{abc[(r * 2) + 1]}{(i * n_comp) + n_comp + 1}']
        
                    condition = b.iloc[0, 0]
                    same_room_answer = b.iloc[1, 1]

                    # Retrieve speakers names
                    speakers = b.iloc[2:, 0].str.split(' - ', expand=True)[1].tolist()

                    if condition == 'NONE':
                        if same_room_answer == 'Yes':
                            # [TRUE NEGATIVE] Correctly answered that they are in the same room
                            # CASE 1
                            df.loc[len(df.index)] = [subject_matlab_id, subject_progressive_id, subject_hrtf,
                                                     complexity_n, room, condition, same_room_answer, 1,
                                                     'NONE', 'NONE', 'NONE', 'NONE', 'NONE', 1,
                                                     ilocnan(speakers, 0), ilocnan(speakers, 1), ilocnan(speakers, 2),
                                                     ilocnan(speakers, 3),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 0)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 1)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 2)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 3))]

                        else:
                            # [FALSE NEGATIVE] Incorrectly answered that they are in the same room
                            position_ans, speaker_ans = b.iloc[2:, ].dropna().iloc[0, 0].split(' - ')

                            # CASE 2
                            df.loc[len(df.index)] = [subject_matlab_id, subject_progressive_id, subject_hrtf,
                                                     complexity_n, room, condition, same_room_answer, 0,
                                                     'NONE', 'NONE', 'NONE', speaker_ans, position_ans, 0,
                                                     ilocnan(speakers, 0), ilocnan(speakers, 1), ilocnan(speakers, 2),
                                                     ilocnan(speakers, 3),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 0)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 1)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 2)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 3))]
        
                    else:
                        impostor_row = [l[0].font.italic for l in b_wb].index(True)
                        position_impostor, speaker_impostor = b_wb[impostor_row][0].value.split(' - ')

                        if same_room_answer == 'No':
                            # [TRUE POSITIVE] Correctly answered that there is an impostor
                            row = b.iloc[2:, ].dropna()
                            position_ans, speaker_ans = row.iloc[0, 0].split(' - ')

                            # CASE 3
                            df.loc[len(df.index)] = [subject_matlab_id, subject_progressive_id, subject_hrtf,
                                                     complexity_n, room, condition, same_room_answer, 1,
                                                     speaker_impostor, speakers_sex[speaker_impostor], position_impostor,
                                                     speaker_ans, position_ans, row.iloc[0, 1],
                                                     ilocnan(speakers, 0), ilocnan(speakers, 1), ilocnan(speakers, 2),
                                                     ilocnan(speakers, 3),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 0)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 1)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 2)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 3))]

                        else:
                            # [FALSE POSITIVE] Incorrectly answered that there is an impostor
                            # CASE 4
                            df.loc[len(df.index)] = [subject_matlab_id, subject_progressive_id, subject_hrtf,
                                                     complexity_n, room, condition, same_room_answer, 0,
                                                     speaker_impostor, speakers_sex[speaker_impostor], position_impostor,
                                                     'NONE', 'NONE', 0,
                                                     ilocnan(speakers, 0), ilocnan(speakers, 1), ilocnan(speakers, 2),
                                                     ilocnan(speakers, 3),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 0)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 1)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 2)),
                                                     ilocnan(speakers_sex, ilocnan(speakers, 3))]

    df.to_csv('test_dataframes/experiment.csv')
    preliminary.index.name = 'trial_id_per_subject'
    preliminary.to_csv('test_dataframes/preliminary.csv')
    subject_info.to_csv('test_dataframes/subject_info.csv', index=False)
