import numpy as np


def array_dimensions_match_check(arr_1: np.ndarray, arr_2: np.ndarray, by_row: bool = True):
	if not by_row:
		arr_1 = arr_1.T
		arr_2 = arr_2.T

	if arr_1.shape[0] != arr_2.shape[0]:
		raise Exception("Attention! Input np.ndarray mismatch! "
		                "Dimensions in input are:" + str(arr_1.shape) + ' and ' + str(arr_2.shape))

	return


def enlist_1D_array(arr_1: np.ndarray, arr_2: np.ndarray):
	if arr_1.ndim is 1 and arr_2.ndim is 1:
		arr_1 = np.array([arr_1])
		arr_2 = np.array([arr_2])

	# else:
	# 	raise Exception("Attention! Input np.ndarray are not 1D!"
	# 	                "Dimensions in input are:" + str(arr_1.shape) + ' and ' + str(arr_2.shape))

	return [arr_1, arr_2]


if __name__ == "__main__":
	arr_1D_1 = np.zeros(10)
	arr_1D_2 = np.zeros(10)

	arr_2D_1 = np.stack([np.zeros(10)] * 4)
	arr_2D_2 = np.stack([np.zeros(10)] * 3)

	enlist_1D_array(arr_1D_1, arr_1D_2)
	enlist_1D_array(arr_2D_1, arr_1D_2)
	enlist_1D_array(arr_2D_1, arr_2D_2)

	array_dimensions_match_check(arr_1D_1, arr_1D_2)
	array_dimensions_match_check(arr_2D_1, arr_2D_2)
