import random

from scripts.statistics.survey_generator import *

if __name__ == "__main__":
    subject_idx = 4
    subject_cod = '11'
    subject_hrtf = "152"

    survey_setup = json_load("scripts/statistics/survey_setup.json")
    trial_setups = survey_setup["trial_setups"]
    conditions = [("complexity", "latin"),
                  ("room", "shuffle"),
                  ("conditions", "shuffle"),
                  ("speaker", "shuffle")]

    sg = SurveyGenerator(survey_setup, conditions, subject_idx, len(survey_setup['complexity']))
    randomized_dict = sg.get_randomized_dict()

    for complexity, rooms in randomized_dict.items():
        for room, conditions in rooms.items():
            for condition, speakers in conditions.items():
                conditions[condition] = random.sample(speakers, int(complexity))

    workbook = xlsxwriter.Workbook(f'test_results/0{subject_idx}_{subject_cod}_{subject_hrtf}.xlsx')

    cell_format = workbook.add_format({'border': 1, 'align': 'left'})
    impostor_format = workbook.add_format({'italic': 1, 'border': 1, 'align': 'left', 'bg_color': 'pink'})
    speaker_colors = {"DAVID": 'red', "ALEX": 'cyan', "SUSAN": 'lime', "MARIA": 'yellow'}

    merge_format = workbook.add_format({
        'bold': 1,
        'border': 2,
        'align': 'center',
        'valign': 'vcenter'})

    for c_col, complexity in enumerate(randomized_dict.keys()):
        c_idx = c_col * 2
        comp = int(complexity)
        compsheet = workbook.add_worksheet(f'Complexity_{complexity}')
        compsheet.set_column(0, 7, 17)

        for r_col, room in enumerate(randomized_dict[complexity].keys()):
            conditions = randomized_dict[complexity][room].keys()
            n_conditions = len(conditions)

            r_idx = r_col * 2
            compsheet.merge_range(0,
                                  r_idx,
                                  0,
                                  r_idx + 1,
                                  room,
                                  merge_format)

            for cond_idx, condition in enumerate(conditions):
                cond_row = (cond_idx * (comp + 2)) + 1
                compsheet.merge_range(cond_row,
                                      r_idx,
                                      cond_row,
                                      r_idx + 1,
                                      condition,
                                      merge_format)
                compsheet.write(cond_row + 1, r_idx, "same_room", cell_format)
                compsheet.write_blank(cond_row + 1, r_idx + 1, None, cell_format)

                for i, speaker in enumerate(randomized_dict[complexity][room][condition]):
                    impostor = trial_setups[room][complexity]['impostor'][condition]
                    position = trial_setups[room][complexity]['position'][i]
                    if impostor != position:
                        color = speaker_colors[speaker]
                        speaker_format = workbook.add_format({'border': 1, 'align': 'left', 'bg_color': color})
                        pos_format = speaker_format
                    else:
                        pos_format = impostor_format
                    compsheet.write(cond_row + i + 2, r_idx, f'{position} - {speaker}', pos_format)
                    compsheet.write_blank(cond_row + i + 2, r_idx + 1, None, cell_format)

    questionnaire = workbook.add_worksheet("Questionnaire")

    questionnaire.set_column(0, 6, 17)
    questionnaire.set_column(6, 6, 100)

    rooms = survey_setup['room']
    speakers = survey_setup['speaker']
    conditions = ['Ref', 'HOA_Bin', 'FV']

    latin_size = len(speakers)
    latin_square = latin_squares[str(latin_size)]

    s_idx = subject_idx % latin_size
    subject_offset = s_idx * latin_size
    latin_row = latin_square[subject_offset: subject_offset + latin_size]

    questionnaire.write(0, 0, "Room", merge_format)
    questionnaire.write(0, 1, "Condition", merge_format)
    questionnaire.write(0, 2, "Speaker", merge_format)
    questionnaire.write(0, 3, "Externalization", merge_format)

    questionnaire.write(0, 4, "Question", merge_format)
    questionnaire.write(1, 4, "Age", cell_format)
    questionnaire.write(2, 4, "Sex", cell_format)
    questionnaire.write(3, 4, "Reverberation", cell_format)
    questionnaire.write(4, 4, "VR", cell_format)
    questionnaire.write(5, 4, "Impairment", cell_format)

    questionnaire.write(0, 5, "Answer", merge_format)
    questionnaire.write(0, 6, "Feedback", merge_format)

    for i, speaker in enumerate(latin_row):
        idx = i % len(rooms)

        color = speaker_colors[speakers[speaker]]
        speaker_format = workbook.add_format({'border': 1, 'align': 'left', 'bg_color': color})

        row_idx = i * 3

        questionnaire.write(row_idx + 1, 0, rooms[idx], cell_format)
        questionnaire.write(row_idx + 2, 0, rooms[idx], cell_format)
        questionnaire.write(row_idx + 3, 0, rooms[idx], cell_format)

        conditions = random.sample(conditions, len(conditions))

        questionnaire.write(row_idx + 1, 1, conditions[0], cell_format)
        questionnaire.write(row_idx + 2, 1, conditions[1], cell_format)
        questionnaire.write(row_idx + 3, 1, conditions[2], cell_format)

        questionnaire.write(row_idx + 1, 2, speakers[speaker], speaker_format)
        questionnaire.write(row_idx + 2, 2, speakers[speaker], speaker_format)
        questionnaire.write(row_idx + 3, 2, speakers[speaker], speaker_format)

    workbook.close()
    print(0)
