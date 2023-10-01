import pymatgen.io.vasp.inputs as vaspin
from pymatgen.core.structure import Structure
import subprocess as sp
import os
import numpy as np

# np.random.seed(0)


class perturbedStructure:
    def __init__(self, root_path='../Vaspinput/'):
        self.root_path = root_path
        self.vaspinput = vaspin.VaspInput.from_directory(self.root_path)
        self.initial = self.vaspinput.get('POSCAR').structure
        self.lattice = self.initial.lattice
        self.species = self.initial.species
        self._len = len(self.species)

    def perturbAllSites(self, perturb_strength: float):
        cart_coords = self.initial.cart_coords

        # Random Sampling in a circle with radius R
        r = np.array([perturb_strength * np.sqrt(np.random.rand(self._len))]) # 1x60
        theta = np.array(np.random.rand(self._len) * 2 * np.pi)  # (60,)
        temp = np.concatenate(([np.cos(theta)],[np.sin(theta)], [np.zeros(60)]),axis=0) # 3x60
        # print(r.shape, theta.shape, temp.shape)
        r_vector = np.transpose(r * temp) # 60x3

        # Convert spherical to cartesian
        rand_length = np.linalg.norm(r_vector)
        new_coords = cart_coords + r_vector
        new_structure = Structure(self.lattice, self.species, new_coords, coords_are_cartesian=True)
        return new_structure

    def cp_files_to(self,target_dir,files=None):
        if files is None:
            files = ["INCAR", "KPOINTS", "POTCAR", "run.py"]

        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        for f in files:
            source_file = os.path.join(self.root_path,f)
            target_file = os.path.join(target_dir, f)
            sp.run(['cp',source_file,target_file])


def mytest():
    myPoscar = perturbedStructure()
    new_structure =myPoscar.perturbAllSites(0.2)
    myPoscar.cp_files_to("test/", files=['INCAR', "KPOINTS"])
    new_structure.to(fmt='poscar',filename="test/POSCAR")

if __name__ == '__main__':
    mytest()