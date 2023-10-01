# Data Generation
Prepare your root working directory as following
```
AreaElement/
  ├──source_poscar/
  |        ├──INCAR
  |        ├──POSCAR
  |        ├──KPOINTS
  |        ├──POTCAR
  |        └──run.py  # shell script to submit DFT jobs for running
  ├──AE_root/  # empty folder that stores the perturbed structures
  |     ├──... # auto generate
  |     ...
  
```
---
## 1. Configure Environment
```
pip install -r requirements.txt
```
## 2. Start Generation
Run `AE_processing.py` to generate perturbed structures by
```
python AE_processing.py --log_file=<NAME_OF_JSON_FILE> --num_poscars=<NUM_POSCARS_FOR_EACH_PERTURB_STRENGTH> --perturb_strength=<LIST_OF_PERTURB_STRENGTH> --generate=True
```
For example, you can generate total 100 (25 for each) structures with 4 perturbed strength 2%, 3%, 4%, 5% of the original bond length and save the information into AreaElement.json file by the following

```
python AE_processing.py --log_file="AreaElement.json" --num_poscars=25 --perturb_strength 0.02, 0.03, 0.04, 0.05 --generate=True
```
Or use the short-hand as the following command.
```
python AE_processing.py -lo="AreaElement.json" -np=25 -pstr 0.02 0.03 0.04 0.05 -g=True
```
---
After generation, you will see 
1. structures with [VASP](https://www.vasp.at/) inputs (INCAR, KPOINTS, POCARS, POTCAR) and also a shell script under `AE_root/`
2. sturcture infomation is stored in a json file, e.g.,
```
AreaElement/
  ├──AE_root/  
  |     ├──perturb_0/ # auto generate
  |     ├──perturb_1/
  |     ├──perturb_2/
  |     ...
```
```
{
  "0": {
      "path": "AE_root/perturb_0",
      "source_dir": "source_poscar/",
      "state": "C",
      "perturb_strength": 0.02,
      "system": "BN",
      "total_energy": -527.72762,
      "pressure": "0.20",
      "computing time": "20.413366666666665 min"
    },
    "1": {
      "path": "AE_root/perturb_1",
      "source_dir": "source_poscar/",
      "state": "C",
      "perturb_strength": 0.02,
      "system": "BN",
      "total_energy": -527.69733,
      "pressure": "0.27",
      "computing time": "20.412583333333334 min"
  ...
```
## 3. Submit Jobs for DFT calculations
Prepare your own shell script and store it in `source_poscar/` and submit the jobs for running by
```
python AE_processing.py --submit_to_run=True --num_jobs=<NUM_JOBS>
```
For example, submit 100 jobs by
```
python AE_processing.py --submit_to_run=True --num_jobs=100
```
