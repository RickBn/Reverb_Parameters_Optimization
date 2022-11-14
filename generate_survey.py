from scripts.statistics.survey_generator import *

if __name__ == "__main__":
    subject_idx = 0
    subject_hrtf = "050"

    survey_setup = json_load("scripts/statistics/survey_setup.json")
    trial_setups = survey_setup["trial_setups"]
    conditions = [("complexity", "latin"),
                  ("room", "shuffle"),
                  ("conditions", "shuffle"),
                  ("speaker", "shuffle")]

    sg = SurveyGenerator(survey_setup, conditions, subject_idx)
    randomized_dict = sg.get_randomized_dict()

    for complexity, rooms in randomized_dict.items():
        for room, conditions in rooms.items():
            for condition, speakers in conditions.items():
                conditions[condition] = speakers[:int(complexity)]

    workbook = xlsxwriter.Workbook(f'test_results/Subject_{subject_idx}_{subject_hrtf}.xlsx')
    worksheet = workbook.add_worksheet()

    cell_format = workbook.add_format({'border': 1, 'align': 'left'})
    impostor_format = workbook.add_format({'border': 1, 'align': 'left', 'bg_color': 'pink'})
    speaker_colors = {"DAVID": 'red', "ALEX": 'cyan', "SUSAN": 'lime', "MARIA": 'yellow'}

    worksheet.set_column(0, 7, 17)

    merge_format = workbook.add_format({
        'bold': 1,
        'border': 2,
        'align': 'center',
        'valign': 'vcenter'})

    for c_col, complexity in enumerate(randomized_dict.keys()):
        c_idx = c_col * 2
        comp = int(complexity)
        worksheet.merge_range(0,
                              c_idx,
                              0,
                              c_idx + 1,
                              f'Complexity_{complexity}',
                              merge_format)
        for r_row, room in enumerate(randomized_dict[complexity].keys()):
            conditions = randomized_dict[complexity][room].keys()
            n_conditions = len(conditions)
            r_idx = (r_row * ((comp + 2) * n_conditions)) + (r_row + 1)
            worksheet.merge_range(r_idx,
                                  c_idx,
                                  r_idx,
                                  c_idx + 1,
                                  room,
                                  merge_format)

            for cond_idx, condition in enumerate(conditions):
                cond_row = r_idx + (cond_idx * (comp + 2)) + 1
                worksheet.merge_range(cond_row,
                                      c_idx,
                                      cond_row,
                                      c_idx + 1,
                                      condition,
                                      merge_format)
                worksheet.write(cond_row + 1, c_idx, "same_room", cell_format)
                worksheet.write_blank(cond_row + 1, c_idx + 1, None, cell_format)

                for i, speaker in enumerate(randomized_dict[complexity][room][condition]):
                    impostor = trial_setups[room][complexity]['impostor'][condition]
                    position = trial_setups[room][complexity]['position'][i]
                    if impostor != position:
                        color = speaker_colors[speaker]
                        speaker_format = workbook.add_format({'border': 1, 'align': 'left', 'bg_color': color})
                        pos_format = speaker_format
                    else:
                        pos_format = impostor_format
                    worksheet.write(cond_row + i + 2, c_idx, f'{position} - {speaker}', pos_format)
                    worksheet.write_blank(cond_row + i + 2, c_idx + 1, None, cell_format)

    workbook.close()
    print(0)
