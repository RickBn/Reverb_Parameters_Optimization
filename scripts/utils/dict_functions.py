from typing import Dict


def get_dict_idx_value(d: Dict, idx: int):
	value = list(d.values())[idx]

	return value


def get_dict_idx_key(d: Dict, idx: int):
	key = list(d.keys())[idx]

	return key
