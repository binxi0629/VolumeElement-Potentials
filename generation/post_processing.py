from tqdm import tqdm
import subprocess as sp
import os
import json
import re
import argparse


class JobError(Exception):
    pass


def path2id(job_path:str):
    return re.split('_', job_path)[-1]


def clear_files(file_list:list, job_path):
    for f in file_list:
        if os.path.exists(os.path.join(job_path, f)):
            os.remove(os.path.join(job_path, f))


def update_state(job_path):
    outcar_path = os.path.join(job_path, "OUTCAR")
    if os.path.exists(job_path):  # W: waiting, Q: queueing, R:Running, C: Complete, Err: Error
        if os.path.exists(outcar_path):
            # if os.path.exists(os.path.join(job_path,'res.txt')):
            #     return 'C'
            # else:
            #     return 'R'
            return 'C'
        else:
            return 'W'
    else:
        return 'Err'


def get_results_by_path(job_path):
    os.system(f"cd {job_path} && grep F OSZICAR > res.txt && grep pressure OUTCAR >> res.txt && grep Elapsed OUTCAR >> res.txt")
    res_path = os.path.join(job_path, 'res.txt')
    with open(res_path, 'r') as f:
        lines = f.readlines()

    temp_e = re.split('F=\s', lines[0])[-1]
    energy = float(re.split('\sE0=', temp_e)[0])
    temp_p = re.split('pressure\s=.\s', lines[1])[-1]
    temp = re.split('\skB', temp_p)[0]
    pressure = temp.strip()
    temp_time = re.split('\(sec\):.\s', lines[2])[-1]
    elapsed_time = f"{float(temp_time.strip())/60} min" # unit: min
    return energy, pressure, elapsed_time


def forward(config_json='AreaElement.json', get_res=False):
    with open(config_json, 'r') as jf:
        _records = json.load(jf)

    err_list =[]
    ids = _records.keys()
    for each_i in tqdm(ids, desc="state updating"):
        each_path = _records[each_i]["path"]
        each_state = _records[each_i]["state"]
        if each_state == 'Err':
            err_list.append(each_i)  # record failed jobs to list
        current_state = update_state(each_path)
        if get_res and current_state == 'C':
            try:
                energy, pressure, elapsed_time = get_results_by_path(each_path)
            except JobError:
                print("Job Error: Job running or Job crash")
                err_list.append(each_i)
                _records[each_i].update({
                    "state": "Err"
                })
            else:
                _records[each_i].update({
                    "state": current_state,
                    "total_energy": energy,
                    "pressure": pressure,
                    "computing time": elapsed_time,
                })
                del energy, pressure, elapsed_time
        else:
            _records[each_i].update({
                "state": current_state
             })

    with open(config_json, 'w') as wf:
        json.dump(_records, wf, indent=2)

    with open("ErrorJobs.json", 'w') as ef:
        json.dump({"errorjobs":err_list},ef, indent=2)

    print(f"Error jobs: {err_list}")
    print("Finished: saved error jobs to ErrorJobs.json")


def perturb_strategy_default(num):
    num_poscars=50
    perturb_strength =[0.02, 0.03, 0.04, 0.05, 0.02, 0.03, 0.04, 0.05]
    return perturb_strength[num//50]


def record_from_files(perturb_strategy=perturb_strategy_default,update_mode=True, _record="AreaElement.json", _root_path="AE_root", source_dir="source_poscar/"):
    """
     If the record file (default: AreaElement.json) is unexpected modified or deleted, run this method to recover it.
    :param perturb_strategy:
    :param update_mode: True: append mode; False: write a new json file
    :param _record: default
    :param _root_path: default
    :param source_dir:  default
    :return: None
    """
    if update_mode and os.path.exists(_record):
        with open(_record, 'r') as jf:
            record_dict = json.load(jf)
    else:
        record_dict = {}

    err_list = []

    for files, subdirs, dirs in tqdm(os.walk(_root_path), desc="processing"):
        for sub_d in subdirs:
            record_key = re.split('_', sub_d)[-1]

            each_path = os.path.join(_root_path, sub_d)
            current_state = update_state(each_path)
            if current_state == 'C':
                try:
                    energy, pressure, elapsed_time = get_results_by_path(each_path)
                except JobError:
                    print("Job Error: Job running or Job crash")
                    err_list.append(record_key)

                    record_dict.update({f"{record_key}":{
                        'path': each_path,
                        'source_dir': source_dir,
                        'state': "Err",  # W: waiting, Q: queueing, R:Running, C: Complete, Err: Error
                        'perturb_strength': perturb_strategy(int(record_key)),  # FIXME: modify this when using other perturbation strategy
                        'system': "BN"  # by default
                    }})
                else:
                    record_dict.update({f"{record_key}": {
                        'path': each_path,
                        'source_dir': source_dir,
                        'state': current_state,  # W: waiting, Q: queueing, R:Running, C: Complete, Err: Error
                        'perturb_strength': perturb_strategy(int(record_key)),
                        # FIXME: modify this when using other perturbation strategy
                        'system': "BN",  # by default
                        "total_energy": energy,
                        "pressure": pressure,
                        "computing time": elapsed_time,
                    }})

                    del energy, pressure, elapsed_time
            else:
                record_dict.update({f"{record_key}": {
                    'path': each_path,
                    'source_dir': source_dir,
                    'state': current_state,  # W: waiting, Q: queueing, R:Running, C: Complete, Err: Error
                    'perturb_strength': perturb_strategy(int(record_key)),
                    # FIXME: modify this when using other perturbation strategy
                    'system': "BN"  # by default
                }})

    with open(_record, 'w') as wf:
        json.dump(record_dict, wf, indent=2)

    with open("ErrorJobs.json", 'w') as ef:
        json.dump({"errorjobs":err_list},ef, indent=2)

    print(f"Error jobs: {err_list}")
    print("Finished: saved error jobs to ErrorJobs.json")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_file", type=str, default="AreaElement.json", nargs="?",
                        help="File name that store structures info")
    parser.add_argument("--get_result", type=bool, default=True, nargs="?",
                        help="Set to True if record the calculated results")
    parser.add_argument("--update_mode", type=bool, default=True, nargs="?",
                        help="Set to True, if update current state")

    parser.add_argument("--record", type=bool, default=False, nargs="?",
                        help="Manually update structure state")

    args = parser.parse_args()

    forward(args.log_file, get_res=args.get_result)

    if args.record:
        record_from_files(update_mode=args.update_mode, _record=args.log_file)