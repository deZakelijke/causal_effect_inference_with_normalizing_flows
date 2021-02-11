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
            if not tag.startswith("metrics/test"):
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
        score = []
        for summ_dict in summ_dict_list:
            score.append(summ_dict[key][-1])
        print(key)
        print(f"Mean: {tf.reduce_mean(score).numpy():.3f},\t"
                f"Std: {tf.math.reduce_std(score).numpy():.3f}")


def process_data(log_path):
    all_paths = get_all_summaries(log_path)
    summ_dict_list = []
    for path in all_paths:
        summ_dict_list.append(collect_summary_dict(path))
    if not summ_dict_list:
        return
    print(f"\n{log_path}")
    get_mean_and_std(summ_dict_list)
    print()

log_paths = ["~/logs/CEVAE/SPACE/", "~/logs/CEVAE/TWINS"]
for log_path in log_paths:
    process_data(log_path)
