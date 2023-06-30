import os
import yaml

actual_params_path = 'audio/input/chosen_rirs/stereo/'
tuned_params_path = 'audio/params/stereo/'

for folder in os.listdir(actual_params_path):
    if folder.startswith("SDN"):
        with open(os.path.join(actual_params_path, folder, 'parameters.yml'), "r") as stream:
            try:
                actual_params = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                print(exc)

        positions = list(actual_params.keys())

        tuned_params = dict()
        losses = []

        for pos in positions:
            with open(os.path.join(tuned_params_path, folder, pos, 'SDN', 'SDN.yml'), "r") as stream:
                try:
                     dict_raw = yaml.safe_load(stream)
                     tuned_params[pos] = dict_raw['parameters']

                     losses.append(dict_raw['loss_end_value'])
                except yaml.YAMLError as exc:
                    print(exc)


