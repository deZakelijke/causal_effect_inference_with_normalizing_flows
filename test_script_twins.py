import os

BATCH_SIZE = 32
DATASET = "TWINS"
EPOCHS = 200
EXP_NAME = "test_script"
FEATURE_MAPS = 200
LEARNING_RATE = 0.00001
LOG_STEPS=20
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
    
