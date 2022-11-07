import random
import xlsxwriter

from scripts.utils.json_functions import json_load
from scripts.utils.plot_functions import *

from typing import Dict, List, Tuple

latin_size = 3
latin_square = np.array([1, 0, 2,
                         0, 2, 1,
                         0, 2, 1,
                         2, 1, 0])

# latin_size = 4
# latin_square = np.array([1, 3, 0, 2,
#                          3, 0, 2, 1,
#                          0, 2, 1, 3,
#                          2, 1, 3, 0])


class SurveyGenerator:
    def __init__(self, setup_dict: Dict[str, List] = None,
                 nested_conditions: List[Tuple[str, str]] = None,
                 subject_id: int = 0):

        self.setup_dict = setup_dict
        self.nested_conditions = nested_conditions

        for condition in nested_conditions:
            if condition[1] not in ["latin", "shuffle"]:
                raise Exception("Attention! "
                                "The randomization type for each condition must be a string = [latin, shuffle]).")

        idx = subject_id % latin_size
        subject_offset = idx * latin_size
        self.latin_row = latin_square[subject_offset: subject_offset + latin_size]

    def get_randomized_dict(self):
        randomized_dict = self.nested_randomization()
        return randomized_dict

    def nested_randomization(self, start: int = 0):
        idx = start
        output_dict = {}
        shuffled_conditions = self.randomize_conditions(self.nested_conditions[idx])

        if idx < len(self.nested_conditions) - 1:
            for condition in shuffled_conditions:
                output_dict[condition] = self.nested_randomization(idx + 1)
        else:
            return shuffled_conditions

        return output_dict

    def randomize_conditions(self, condition: Tuple[str, str]):
        condition_name = condition[0]
        random_type = condition[1]

        if random_type is "latin":
            return self.randomize_latin(condition_name)
        elif random_type is "shuffle":
            return self.randomize_shuffle(condition_name)
        else:
            raise Exception("Attention! "
                            "The randomization type for each condition must be a string = [latin, shuffle]).")

    def randomize_latin(self, condition_name: str):
        condition_list = self.setup_dict[condition_name]
        if len(condition_list) != latin_size:
            raise Exception("Attention! Mismatch between chosen latin square size and condition list.")

        shuffled_list = []
        for idx in self.latin_row:
            shuffled_list.append(str(condition_list[idx]))

        return shuffled_list

    def randomize_shuffle(self, condition_name: str):
        condition_list = self.setup_dict[condition_name]
        shuffled_list = random.sample(condition_list, len(condition_list))

        return shuffled_list


if __name__ == "__main__":
    subject_idx = 0
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

    workbook = xlsxwriter.Workbook('Survey.xlsx')
    worksheet = workbook.add_worksheet()

    cell_format = workbook.add_format({'border': 1, 'align': 'left'})
    impostor_format = workbook.add_format({'border': 1, 'align': 'left', 'bg_color': 'purple'})
    speaker_colors = {"DAVID": 'red', "RICHARD": 'cyan', "SUSAN": 'lime', "MARIA": 'yellow'}

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
                    worksheet.write(cond_row + i + 2, c_idx, f'{position}-{speaker}', pos_format)
                    worksheet.write_blank(cond_row + i + 2, c_idx + 1, None, cell_format)

    workbook.close()
    print(0)
