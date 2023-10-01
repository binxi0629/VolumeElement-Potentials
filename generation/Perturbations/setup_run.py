import pymatgen.io.vasp.inputs as vaspin
from pymatgen.core.structure import Structure
import subprocess as sp
import os
import numpy as np

debug = True # safety lock, doesnt qsub

def write_cleanup(para_n):
    with open("cleanup.sh") as sc:
        script = sc.read()
        script = script.format(para=para_n)
    with open("clean.sh", 'w') as sc:
        sc.write(script)

def run_each(dirname,debug=False):
    sp.run(['cp','run.sh',dirname])
    if debug:
        return
    os.chdir(f"./{dirname}")
    sp.run(['qsub','run.sh'])
    os.chdir("../..")

def io_setup(vaspinput,para,para_n):
    dirname = f"{para_n}{para}"
    vaspinput.write_input(dirname)
    run_each(dirname,debug=debug)

def setup_run(site_index : int ,trials : int,perturb_strength : float,write_clean : bool):
    vaspinput = vaspin.VaspInput.from_directory('../Vaspinput/')
    initial = vaspinput.get('POSCAR').structure
    lattice = initial.lattice
    species = initial.species
    with open("Randlogs",'w+') as log:
        for index in range(trials):
            cart_coords = initial.cart_coords
            # Random Sampling in a circle with radius R
            r = perturb_strength * np.sqrt(np.random.rand())
            theta = np.random.rand() * 2 * np.pi
            r_vector = r * np.array([np.cos(theta),np.sin(theta),0])
            # Convert spherical to cartesian 
            rand_length = np.linalg.norm(r_vector)
            cart_coords[site_index] += r_vector
            new = Structure(lattice,species,cart_coords,coords_are_cartesian=True)
            vaspinput['POSCAR'] = new

            io_setup(vaspinput,f"Trial{index}",f"ATM{site_index}")
            arrstr = np.char.mod('%f', r_vector)
            arrstr = "\t".join(arrstr)
            # Index of the run, length , 1x3 array ( Z is always 0 )
            log.write(f"{index}\t{rand_length}\t{arrstr}\n")
    if write_clean:
        write_cleanup(f"ATM{site_index}")

if __name__ == "__main__":
    setup_run(5,20,1,write_clean=True)
