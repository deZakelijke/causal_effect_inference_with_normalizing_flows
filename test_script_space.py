import os

BATCH_SIZE = 16
DATASET = "TWINS"
EPOCHS = 100
EXP_NAME = "test_script_original_data_4_flows"
FEATURE_MAPS = 32
LEARNING_RATE = 0.0001
LOG_STEPS=10
N_FLOWS = 4


MODELS = ["TARNET", "CEVAE", "PlanarFlow", "RadialFlow", "SylvesterFlow", "NCF"]
FLOW_TYPE = ["AffineCoupling", "NLSCoupling"]

for model in MODELS:
    command = "python3.7 src/main.py "\
              f"--batch_size {BATCH_SIZE} "\
              f"--dataset {DATASET} "\
              f"--epochs {EPOCHS} "\
              f"--experiment_name {EXP_NAME} "\
              f"--feature_maps {FEATURE_MAPS} "\
              f"--flow_type AffineCoupling "\
              f"--learning_rate {LEARNING_RATE} "\
              f"--model {model} "\
              f"--n_flows {N_FLOWS} "
    os.system(command)

command = "python3.7 src/main.py "\
          f"--batch_size {BATCH_SIZE} "\
          f"--dataset {DATASET} "\
          f"--epochs {EPOCHS} "\
          f"--experiment_name {EXP_NAME} "\
          f"--feature_maps {FEATURE_MAPS} "\
          f"--flow_type NLSCoupling "\
          f"--learning_rate {LEARNING_RATE} "\
          f"--model {model} "\
          f"--n_flows {N_FLOWS} "
os.system(command)
    
