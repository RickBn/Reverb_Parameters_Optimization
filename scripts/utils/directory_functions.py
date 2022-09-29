import os


def directory_filter(input_path: str):
	return [file for file in os.listdir(input_path) if not os.path.isdir(f'{input_path}/{file}')]
