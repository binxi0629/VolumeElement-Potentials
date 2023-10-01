import argparse
import json
import os.path
import os
from Perturbations.perturbed_structure import perturbedStructure

_root_path = "AE_root"


def gen_poscars(num_poscars, source_dir, last_start, perturb_strength, cryst_system="BN", record_dict=None):

    if record_dict is None:
        _record = {}
    else:
        _record = record_dict

    for i in range(last_start, num_poscars+last_start):
        trg_dir = os.path.join(_root_path,f"perturb_{i}")
        myPoscar = perturbedStructure(source_dir)
        new_structure = myPoscar.perturbAllSites(perturb_strength)
        myPoscar.cp_files_to(trg_dir, files=['INCAR', "KPOINTS", "POTCAR", "run.sh"])
        filename = os.path.join(trg_dir, "POSCAR")
        new_structure.to(fmt='poscar', filename=filename)

        _record.update({i: {
            'path': trg_dir,
            'source_dir': source_dir,
            'state': 'W',  # W: waiting, Q: queueing, R:Running, C: Complete, Err: Error
            'perturb_strength': perturb_strength,
            'system': cryst_system
        }})
    return _record


def submitjob_by_path(record_dict:dict, jobidx:int):
    job_path = record_dict[f'{jobidx}']['path']
    os.system(f"cd {job_path} && qsub -N {jobidx} run.sh")
    record_dict[f'{jobidx}'].update({"state": 'Q'})


def gen(num_poscars:int, perturb_strength:list, last_start:int, _record="AreaElement.json"):
    source_dir = "source_poscar/"

    if os.path.exists(_record):
        with open(_record, 'r') as jsonf:
            record_dict =json.load(jsonf)
    else:
        record_dict = {}
    
    if not os.path.exists(_root_path):
        os.makedirs(_root_path)

    for i in range(len(perturb_strength)):
        record_dict = gen_poscars(num_poscars=num_poscars,
                                  source_dir=source_dir,
                                  perturb_strength=perturb_strength[i],
                                  last_start=num_poscars * i+last_start,
                                  cryst_system='BN',
                                  record_dict=record_dict)

    with open('AreaElement.json', 'w') as jf:
        json.dump(record_dict, jf, indent=2)

    print("Saved records to AreaElement.json")


def submit_jobs(lastsubmitted_job_idx:int,num_jobs, job_list=None, _record="AreaElement.json", jobs_log="jobs_log.json"):

    with open(_record, 'r') as jf:
        record_dict = json.load(jf)

    if job_list is not None:
        jobs = job_list[0:num_jobs] if num_jobs < len(job_list) else job_list
    else:
        jobs = list(range(lastsubmitted_job_idx+1, num_jobs+lastsubmitted_job_idx+1))

    for each_job in jobs:
        submitjob_by_path(record_dict,each_job)

    with open('AreaElement.json', 'w') as w_jf:
        json.dump(record_dict, w_jf, indent=2)


if __name__ == '__main__':
    parse = argparse.ArgumentParser()
    parse.add_argument("-lo","--log_file", type=str, default="AreaElement.json", nargs="?",
                       help="File name to store the generation info")
    parse.add_argument("-np","--num_poscars", type=int, default=50, nargs="?",
                       help="number of structures to generate for each perturbed strength, "
                            "Note: total structures=num_posars * len(perturb_strength)")
    parse.add_argument("-pstr","--perturb_strength", type=float, default=[0.02, 0.03, 0.04, 0.05], nargs="+",
                       help="list of perturbed strength, e.g., 0.01 means 1% of current bond length")
    parse.add_argument('-istart',"--last_start", type=int, default=0, nargs="?",
                       help="If you want to generate more structures, "
                            "set this value as the id of the last generated one")
    parse.add_argument('-istop',"--last_stop", type=int, default=-1, nargs="?",
                       help="The id of last submitted job")
    parse.add_argument('-nj',"--num_jobs", type=int, default=100, nargs="?",
                       help="Number of jobs to submitted")
    parse.add_argument('-ls',"--job_list", type=list, default=None, nargs="?",
                       help="Specified job lists to be submitted")

    parse.add_argument('-sub',"--submit_to_run", type=bool, default=False, nargs="?",
                       help="Set to True if run the calculations")
    parse.add_argument('-g',"--generate", type=bool, default=False, nargs="?",
                       help="Set to True if generate structures")

    args = parse.parse_args()
    if args.generate:
        gen(
            _record=args.log_file,
            num_poscars=args.num_poscars,
            perturb_strength=args.perturb_strength,
            last_start=args.last_start
        )

    if args.submit_to_run:
        submit_jobs(
            lastsubmitted_job_idx=args.last_stop,
            num_jobs=args.num_jobs,
            job_list=args.job_list
        )
