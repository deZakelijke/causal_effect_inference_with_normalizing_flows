import os
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator


def get_all_summaries(log_dir):
    all_paths = []
    for (dirpath, dirnames, filenames) in os.walk(log_dir):
        new_paths = [os.path.join(dirpath, file) for file in filenames]
        all_paths += [path for path in new_paths if path.endswith(".v2")]
    return all_paths


def collect_summary_dict(path):
    summ = list(summary_iterator(path))
    summ_dict = {}
    
    for item in summ[1:]:
        if item.summary.value[0].metadata.plugin_data.plugin_name == "scalars":
            tag = item.summary.value[0].tag
            if (not tag.startswith("metrics/test")) and\
               (not tag.startswith("metrics/train/loss")):
                continue
            data = tf.make_ndarray(item.summary.value[0].tensor)
            if tag in summ_dict:
                summ_dict[tag].append(data)
            else:
                summ_dict[tag] = [data]
    for key in summ_dict:
        summ_dict[key] = tf.stack(summ_dict[key])

    return summ_dict


def get_mean_and_std(summ_dict_list):
    for key in summ_dict_list[0]:
        min_score = []
        mean_score = []
        last_score = []
        for summ_dict in summ_dict_list:
            min_score.append(tf.reduce_min(summ_dict[key]))
            mean_score.append(tf.reduce_mean(summ_dict[key]))
            last_score.append(summ_dict[key][-1])
        print(key)
        print("Minimum")
        print(f"Mean: {tf.reduce_mean(min_score).numpy():.4f},\t"
                f"Std: {tf.math.reduce_std(min_score).numpy():.4f}")
        print("Mean")
        print(f"Mean: {tf.reduce_mean(mean_score).numpy():.4f},\t"
                f"Std: {tf.math.reduce_std(mean_score).numpy():.4f}")
        print("Last")
        print(f"Mean: {tf.reduce_mean(last_score).numpy():.4f},\t"
                f"Std: {tf.math.reduce_std(last_score).numpy():.4f}")


def process_data(log_path, prefix):
    all_paths = get_all_summaries(prefix + log_path)
    summ_dict_list = []
    for path in all_paths:
        summ_dict_list.append(collect_summary_dict(path))
    if not summ_dict_list:
        return
    print(f"\n{log_path}")
    get_mean_and_std(summ_dict_list)
    print()


prefix = "/home/mdegroot/logs/"
log_paths_space = ["TARNET/SPACE/", "CEVAE/SPACE/", "PlanarFlow/SPACE/",
                   "RadialFlow/SPACE/", "SylvesterFlow/SPACE/",
                   "NCF/AffineCoupling/SPACE/", "NCF/NLSCoupling/SPACE"]
for log_path in log_paths_space:
    process_data(log_path, prefix)

log_paths_twins = ["TARNET/TWINS/", "CEVAE/TWINS/", "PlanarFlow/TWINS/",
                   "RadialFlow/TWINS/", "SylvesterFlow/TWINS/",
                   "NCF/AffineCoupling/TWINS/", "NCF/NLSCoupling/TWINS/"]
for log_path in log_paths_twins:
    process_data(log_path, prefix)
