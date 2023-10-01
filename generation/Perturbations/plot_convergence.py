from pymatgen.io.vasp.outputs import Vasprun
from setup_run import para_name , para_arr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

para_list = []
energy_list = []
fname = f"./graphs/{para_name}_convergence"

# Anti-Overwrite Block
counter = 1
fname_count = fname
while os.path.exists(fname_count + '.png'):
    fname_count = fname + str(counter)
    counter += 1
fname = fname_count + '.png'

# Gathering loop, vectorizable
for para in para_arr:
    dirname = f"./{para_name}{para}/"
    output = Vasprun(dirname + 'vasprun.xml')
    para_list.append(para)
    energy_list.append(float(output.final_energy))

# Plotting graph to fname
plt.plot(para_list,energy_list)
plt.savefig(fname)
print(para_list)
print(energy_list)
