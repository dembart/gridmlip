import numpy as np
from tqdm import tqdm
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator



def read_cfg(filename):

    """
    Read .cfg file format used in MLIP package

    Parameters
    ----------
    file: str
        path to the file
    
    Returns
    ----------
    list of ase's Atoms object with a SinglePointCalculator
    """
    with open(filename, 'r') as f:
        text = f.readlines()
    configurations = []
    for i_s, i_e in tqdm(zip(np.where(np.array(text) == 'BEGIN_CFG\n')[0], np.where(np.array(text) == 'END_CFG\n')[0])):
        subtext = text[i_s:i_e]
        grade = None
        results = {}
        for i, line in enumerate(subtext):
            if 'Size' in line:
                size = int(subtext[i+1])
            if 'Supercell' in line:
                x = subtext[i+1]
                y = subtext[i+2]
                z = subtext[i+3]
                cell = np.array([x.split(), y.split(), z.split()], dtype = float)
            if 'AtomData' in line:
                pos, forces, numbers = [], [], []
                for n in range(size):
                    if len(subtext[i+n+1].split()) == 8:
                        id_, num, pos_x, pos_y, pos_z, f_x, f_y, f_z = subtext[i+n+1].split()
                        pos.append(np.array([pos_x, pos_y, pos_z], dtype = float))
                        forces.append(np.array([f_x, f_y, f_z], dtype = float))
                        numbers.append(num)
                    elif len(subtext[i+n+1].split()) == 5:
                        id_, num, pos_x, pos_y, pos_z = subtext[i+n+1].split()
                        pos.append(np.array([pos_x, pos_y, pos_z], dtype = float))
                        forces.append(np.array([None, None, None], dtype = float))
                        numbers.append(num)
                results.update({'forces': forces})
            if 'MV' in line: 
                grade = float(line.split()[-1])
            if 'Energy' in line:
                energy =  float(subtext[i+1].split()[-1])
                results.update({'energy': energy})
            if 'PlusStress' in line:
                stress = np.array(subtext[i+1].split(), dtype = float)
                results.update({'stress': stress})
        atoms = Atoms(cell = cell, positions = np.array(pos), numbers = numbers, pbc = True)
        calc = SinglePointCalculator(atoms, **results)
        atoms.calc = calc
        configurations.append(atoms)
    return configurations


def write_cfg(filename, atoms_list):
    
    """
    Write list of atoms into .cfg file used in MLIP package

    Parameters
    ----------
    filename: str
        path to the file
    """

    with open(filename, 'w') as text:
        for atoms in tqdm(atoms_list):
            
            text.write('BEGIN_CFG\n')

            # Size
            text.write(' Size\n')
            text.write('{:6}'.format(len(atoms)) + '\n')
            
            # Supercell
            box = np.array(atoms.cell)
            text.write(' Supercell\n')
            text.write(''.join(map('{:16.6f}'.format, box[0, :])) + '\n')
            text.write(''.join(map('{:16.6f}'.format, box[1, :])) + '\n')
            text.write(''.join(map('{:16.6f}'.format, box[2, :])) + '\n')

            
            keys = []
            keys.append('ids')
            ids =  np.arange(1, len(atoms) + 1)
            
            keys.append('type')
            types = atoms.numbers
            
            keys.append('cartes_x')
            cartes_x = atoms.positions[:, 0]
            keys.append('cartes_y')
            cartes_y = atoms.positions[:, 1]
            keys.append('cartes_z')
            cartes_z = atoms.positions[:, 2]

            # forces
            try:
                forces = atoms.get_forces()
                keys.append('fx')
                fx = forces[:, 0]
                keys.append('fy')
                fy = forces[:, 1]
                keys.append('fz')
                fz = forces[:, 2]
                text.write(' AtomData:  id type       cartes_x      cartes_y      cartes_z           fx          fy          fz\n')
                data = np.vstack([ids, types, cartes_x, cartes_y, cartes_z, fx, fy, fz]).T
            except:
                text.write(' AtomData:  id type       cartes_x      cartes_y      cartes_z\n')
                data = np.vstack([ids, types, cartes_x, cartes_y, cartes_z]).T
            
            for i in range(len(data)):
                row = list(data[i, :])
                row[0] = int(row[0])
                row[1] = int(row[1])
                if len(row) == 5:
                    text.write('{:13} {:2}     {:>13.6f}      {:>6.6f}     {:>9.6f}'.format(*row) + '\n')
                elif len(row) == 8:
                    text.write('{:13} {:2}     {:>13.6f}      {:>6.6f}      {:>9.6f}    {:>9.6f}   {:>9.6f}   {:>9.6f}\n'.format(*row))

            try:
                energy = atoms.get_potential_energy()
                text.write(' Energy\n')
                text.write('{:>13.6f}'.format(energy) + '\n')
            except:
                pass

            try:
                stress = atoms.get_stress()
                text.write(' PlusStress:  xx          yy          zz          yz          xz          xy\n')
                text.write(' {:>15.6f}        {:>4.6f}        {:>4.6f}        {:>4.6f}        {:>4.6f}        {:>4.6f}'.format(*stress) + '\n')
            except: 
                pass
            text.write(' Feature   EFS_by no_info\n')
            text.write('END_CFG\n\n')
