import numpy as np
from ase.atoms import Atoms
from itertools import islice
from typing import Dict, List, Set, Tuple, Union

class ConfigASE(object):
    def __init__(self):
        self.info = {}
        self.cell = None
        self.pbc = np.array([False, False, False])
        self.atoms = []
        self.positions = []
        self.symbols = []
    def __len__(self):
        return len(self.atoms)
    def get_positions(self):
        return self.positions
    def get_chemical_symbols(self):
        return self.symbols
    def create(self, n_atoms, fs):
        # Read atoms
        self.positions = []
        self.symbols = []
        for i in range(n_atoms):
            ln = fs.readline()
            ln = ln.split()
            name = ln[0]
            pos = list(map(float, ln[1:4]))
            pos = np.array(pos)
            self.positions.append(pos)
            self.symbols.append(name)
        self.positions = np.array(self.positions)
        return

def read_xyz(config_file,
            index=':'):
        species={'C'}
        atom_list = []
        mol_list = []
        num_list = []
        ifs = open(config_file, 'r')
        while True:
            header = ifs.readline().split()
            if header != []:
                assert len(header) == 1
                n_atoms = int(header[0])
                tmp = ifs.readline()
                num_list.append(n_atoms)
                config = ConfigASE()
                config.create(n_atoms, ifs)
                atom_list.append(config.get_chemical_symbols())
                atoms = set(config.get_chemical_symbols())
                if (atoms.issubset(species)==False):
                    species = species.union(atoms)
                xyz = config.get_positions()
                mol = Atoms(symbols=config.get_chemical_symbols(), positions= xyz)
                mol_list.append(mol)
            else: break
        return mol_list, num_list, atom_list, species

def split_by_lengths(seq, num):
    out_list = []
    i=0
    for j in num:
        out_list.append(seq[i:i+j])
        i+=j
    return out_list
