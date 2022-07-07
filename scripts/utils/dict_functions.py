from typing import Dict


def get_dict_idx_value(d: Dict, idx: int):
	value = list(d.values())[idx]

	return value


def get_dict_idx_key(d: Dict, idx: int):
	key = list(d.keys())[idx]

	return key


def exclude_keys(d: Dict, keys):
	return {x: d[x] for x in d if x not in keys}
