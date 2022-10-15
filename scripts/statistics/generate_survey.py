import os

import numpy as np
import pandas as pd
import math
import random

from scripts.utils.json_functions import json_load
from scripts.utils.plot_functions import *

from typing import Dict, List, Tuple

latin_size = 4
latin_square = np.array([1, 3, 0, 2,
                         3, 0, 2, 1,
                         0, 2, 1, 3,
                         2, 1, 3, 0])


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

		self.randomized_dict = {}

	def randomize_conditions(self):
		for condition in nested_conditions:
			condition_name = condition[0]
			random_tpye = condition[1]

			if random_tpye is "latin":
				self.randomize_latin(condition_name)
			elif random_tpye is "shuffle":
				self.randomize_shuffle(condition_name)

		return self.randomized_dict

	def randomize_latin(self, condition_name: str):
		condition_list = self.setup_dict[condition_name]
		if len(condition_list) != latin_size:
			raise Exception("Attention! Mismatch between chosen latin square size and condition list.")

		shuffled_list = []
		for idx in self.latin_row:
			shuffled_list.append(condition_list[idx])

		self.randomized_dict[condition_name] = shuffled_list

	def randomize_shuffle(self, condition_name: str):
		shuffled_list = self.setup_dict[condition_name]
		random.shuffle(shuffled_list)

		self.randomized_dict[condition_name] = shuffled_list


if __name__ == "__main__":
	id = 0
	survey_setup = json_load("scripts/statistics/survey_setup.json")
	nested_conditions = [("complexity", "latin"),
	                     ("room", "shuffle"),
	                     ("conditions", "shuffle"),
	                     ("speaker", "shuffle")]

	sg = SurveyGenerator(survey_setup, nested_conditions, id)
	sg.randomize_conditions()
