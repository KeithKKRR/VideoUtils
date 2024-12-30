import argparse
import json
import numpy as np
import subprocess
import os

def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json-file', type=str, required=True)
    parser.add_argument('--proc-per-gpu', type=int, default=5)
    parser.add_argument('--gpu-ids', nargs='+', type=int, required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args =arg_parse()
    total_num_proc = min(os.cpu_count(), args.proc_per_gpu * len(args.gpu_ids))
    processes = list()
    task_json_list = list()

    with open(args.json_file, 'r') as f:
        all_data = json.load(f)

    sep = list(np.round(np.linspace(0, len(all_data), total_num_proc + 1)).astype(np.int32))
    for i in range(total_num_proc):
        task_data = all_data[sep[i]:sep[i + 1]]
        gpu_idx = args.gpu_ids[i // args.proc_per_gpu]
        proc_idx = i % args.proc_per_gpu
        if len(task_data) == 0:
            continue

        task_json = f'tmp_gpu_{gpu_idx}_proc_{proc_idx}.json'
        task_json_list.append(task_json)
        with open(task_json, 'w') as task_f:
            json.dump(task_data, task_f, indent=4)

        command = f'CUDA_VISIBLE_DEVICES={gpu_idx} python my_video_swap.py --json-file {task_json}'
        process = subprocess.Popen(command, shell=True)
        processes.append(process)

    for process in processes:
        process.wait()

    for task_json in task_json_list:
        os.remove(task_json)





