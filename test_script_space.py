import subprocess
import sys

BATCH_SIZE = 16
DATASET = "SPACE"
DATASET_PATH = "datasets/SPACE/test_second_set_n_obj_"
EPOCHS = 100
EXP_NAME = "test_script_second_data_2_flows_fmaps_16"
EXP_NAME = "test_script_second_data_4_flows"
FEATURE_MAPS = 32
LEARNING_RATE = 0.0001
LOG_STEPS=10
N_FLOWS = 4
LOG_DIR = "logs/"
N_SAMPLES = 20
MODE = "test"
N_OBJ = 8


MODELS = ["TARNET", "CEVAE", "PlanarFlow", "RadialFlow", "SylvesterFlow", "NCF"]
# MODELS = ["SylvesterFlow"]
FLOW_TYPE = ["AffineCoupling", "NLSCoupling"]

for model in MODELS:
    # if model == "SylvesterFlow":
    #     n_samples = 20
    # else:
    #     n_samples = 100
    command = "python3.7 src/main.py "\
              f"--batch_size {BATCH_SIZE} "\
              f"--dataset {DATASET} "\
              f"--debug "\
              f"--epochs {EPOCHS} "\
              f"--experiment_name {EXP_NAME} "\
              f"--feature_maps {FEATURE_MAPS} "\
              f"--flow_type AffineCoupling "\
              f"--learning_rate {LEARNING_RATE} "\
              f"--model {model} "\
              f"--mode {MODE} "\
              f"--n_flows {N_FLOWS} "\
              f"--n_samples {N_SAMPLES} "\
              f"--path_dataset {DATASET_PATH}{N_OBJ}_ "
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    stdout = str(stdout, 'utf-8')
    stderr = str(stderr, 'utf-8')
    with open(f"{LOG_DIR}SPACE_stdout_{EXP_NAME}_{MODE}_n_obj_{N_OBJ}.txt", "a") as f:
        f.write(f"\n\n{stdout}\n")
    with open(f"{LOG_DIR}SPACE_stderr_{EXP_NAME}_{MODE}_n_obj_{N_OBJ}.txt", "a") as f:
        f.write(f"{model}\n")
        f.write(f"\n\n{stderr}\n")

# sys.exit(0)

command = "python3.7 src/main.py "\
          f"--batch_size {BATCH_SIZE} "\
          f"--dataset {DATASET} "\
          f"--debug "\
          f"--epochs {EPOCHS} "\
          f"--experiment_name {EXP_NAME} "\
          f"--feature_maps {FEATURE_MAPS} "\
          f"--flow_type NLSCoupling "\
          f"--learning_rate {LEARNING_RATE} "\
          f"--model {model} "\
          f"--mode {MODE} "\
          f"--n_flows {N_FLOWS} "\
          f"--n_samples {N_SAMPLES} "\
          f"--path_dataset {DATASET_PATH}{N_OBJ}_ "
process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
stdout = str(stdout, 'utf-8')
stderr = str(stderr, 'utf-8')
with open(f"{LOG_DIR}SPACE_stdout_{EXP_NAME}_{MODE}_n_obj_{N_OBJ}.txt", "a") as f:
    f.write(f"\n\n{stdout}\n")
with open(f"{LOG_DIR}SPACE_stderr_{EXP_NAME}_{MODE}_n_obj_{N_OBJ}.txt", "a") as f:
    f.write(f"{model}\n")
    f.write(f"\n\n{stderr}\n")
