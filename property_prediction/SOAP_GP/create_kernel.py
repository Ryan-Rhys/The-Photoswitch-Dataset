import os
from concurrent import futures 

from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from ase.atoms import Atoms
from itertools import islice
from typing import Dict, List, Set, Tuple, Union
from dscribe.descriptors import SOAP
from dscribe.kernels import REMatchKernel
from sklearn.preprocessing import normalize


def generate_kernel(task, smiles_list):
    """
    Generates SOAP kernel via a series of file writing and format conversions.
    """

    # Write smiles to a .smi file

    with open('SOAP_GP/tmp/'+task+'.smi', 'w') as f:
        for item in smiles_list:
            f.write("%s\n" % item)

    # Generate conformers and write to .sdf file

    write_sdf(task)
    cmd = 'babel -isdf SOAP_GP/tmp/'+task+'.sdf -oxyz SOAP_GP/tmp/'+task+'.xyz' # convert to .xyz file
    os.system(cmd)

    # Read atom positions from .xyz file

    mols, num_list, atom_list, species = read_xyz('SOAP_GP/tmp/'+task+".xyz")
    rcut = 3.0
    sigma = 0.2
    
    soap_gen = SOAP(
        species=species,
        periodic=False,
        rcut=rcut,
        nmax=12,
        lmax=8,
        sigma = sigma,
        sparse=True
    )
    
    # Create soap descriptors

    soap = soap_gen.create(mols)
    soap = normalize(soap, copy=False)
    soap = split_by_lengths(soap, num_list)
    re = REMatchKernel(metric="polynomial", degree=3, gamma=1, coef0=0, alpha=0.5, threshold=1e-6, normalize_kernel=True)

    return re.create(soap).astype(np.float32)

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
        i=0
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
def generateconformations(m):
    m = Chem.AddHs(m)
    AllChem.EmbedMolecule(m)
    m = Chem.RemoveHs(m)
    return m

def write_sdf(task):
    smiles_name = 'SOAP_GP/tmp/'+task+'.smi'
    sdf_name = 'SOAP_GP/tmp/'+task+'.sdf'
    
    max_workers=32
    
    writer = Chem.SDWriter(sdf_name)
    
    suppl = Chem.SmilesMolSupplier(smiles_name, delimiter=',', titleLine=False)
    
    with futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit a set of asynchronous jobs
        jobs = []
        for mol in suppl:
            if mol:
                job = executor.submit(generateconformations, mol)
                jobs.append(job)
    
        # Process the job results (in submission order) and save the conformers.
        for job in jobs:
            mol= job.result()
            writer.write(mol)
    
    writer.close()


