from dwave.system import DWaveSampler, TilingComposite
import numpy as np
import pandas as pd
import dimod
from dimod.generators.random import ran_r
from orang import OrangSampler, OrangSolver

import logging

logger = logging.getLogger('Tiling')
logger.setLevel(logging.DEBUG)


class QPUTile(object):

    def __init__(self, nodes, edges):

        self.nodes = nodes
        self.edges = edges

    def __len__(self):
        return len(self.nodes)

    def get_tile_energy(self, sampleset, **kwargs):
        pass

    #Outpus which cells are in the tile.
    def which_cells(self):
        cell_indices = []
        for n in self.nodes:
            if n%8 == 0:
                cell_indices.append(int(n/8))
        return cell_indices


class RANrTile(QPUTile):

    def __init__(self, nodes, edges, rval):

        super().__init__(nodes, edges)

        self.bqm = ran_r(rval, (self.nodes, self.edges))

        self.orang_sampler = OrangSampler()

    def size(self):
        return len(list(self.bqm.variables))

    def qpu_tile_energy(self, samples, tiling_composite):
        """

        Args:
            samples: SampleSet object
            tiling_composite: TilingComposite

        Returns:
            Array of energies (floats)

        """
        # Because of missing qubits (tile areas that were skipped), we have to relabel the tile bqm with
        # a sequence of variables that 'compresses out' the mising qubits to align the the indices in `samples`
        idx = {v: i for i, v in enumerate(tiling_composite.child.properties['qubits'])
               if v in self.bqm.variables}

        new_bqm = self.bqm.relabel_variables(idx, inplace=False)
        return np.array([new_bqm.energy(sample) for sample in samples.record['sample']])


class Tiling(object):

    def __init__(self, sub_m=4, sub_n=4, t=4, sampler=None, solver='DW_2000Q_2_1', token=None):

        self.sub_m = sub_m
        self.sub_n = sub_n
        self.t = t
        self.solver = solver

        if sampler is None:
            if token:
                self.sampler = DWaveSampler(solver=self.solver, token=token)
            else:
                self.sampler = DWaveSampler(solver=self.solver)

        self.tiling_composite = self.tile_qpu()

        self.bqm = dimod.BQM.empty('SPIN')
        self.null_bqm = dimod.BQM.empty('SPIN')
        self.tiles = {}
        self.var2idx = {}

    def tile_qpu(self):
        """
        Create a tiling according to the topology of given sampler.
        This will conveniently exclude sections on the Chimera graph where a complete tile is not possible due to one
        or more missing qubits.

        Returns:
            TilingComposite

        """
        logger.info('Generating tiling...')
        return TilingComposite(self.sampler, self.sub_m, self.sub_n, self.t)

    def construct_null_bqm(self):
        bqm_vars = set(self.bqm.variables)
        qpu_vars = set(self.tiling_composite.child.properties['qubits'])
        self.null_bqm.update(dimod.BQM(linear={v: 0 for v in qpu_vars.difference(bqm_vars)},
                                       quadratic={}, offset=0, vartype=dimod.SPIN))

    def get_embedding_size(self):
        return len(self.tiling_composite.nodelist)

    def generate_ran_r_tiles(self, k=1):
        """
        Args:
            k: int, RAN-k where k \in [1, inf]

        Returns:
            dimod.BQM
        """
        for i, em in enumerate(self.tiling_composite.embeddings):
            tile_nodes = []
            tile_edges = []
            for u in self.tiling_composite.nodelist:
                # stupid set() does not have any way to 'get' an element without popping it
                # https://stackoverflow.com/questions/59825/how-to-retrieve-an-element-from-a-set-without-removing-it
                ux = next(iter(em[u]))
                tile_nodes.append(ux)
            for u, v in iter([(em[u], em[v]) for u, v in self.tiling_composite.edgelist]):
                ux = next(iter(u))
                vx = next(iter(v))
                tile_edges.append((ux, vx))
            self.tiles[i] = RANrTile(tile_nodes, tile_edges, k)
            self.bqm.update(self.tiles[i].bqm)
            self._update_var_idx(self.tiles[i].bqm)

        # fill out BQM with remaining available qubits but set them to zero.
        # This is easier than pushing them into the BQM after sampling
        self.construct_null_bqm()
        self.bqm.update(self.null_bqm)
        return self.bqm

    def _update_var_idx(self, bqm):
        v2i = {u: i for i, u in zip(bqm.variables, range(len(self.var2idx),
                                                         len(self.var2idx) + self.get_embedding_size()))
               }
        self.var2idx.update(v2i)

    def sample_tiling(self, **kwargs):
        logger.info('Sampling from {}'.format(self.solver))
        return self.tiling_composite.child.sample(self.bqm, **kwargs)

    def compute_energy_stats(self, energies):
        """

        Args:
            energies: ndarray, num_reads energies. (No error checking for empty array/list)

        Returns:
            mean energy, standard deviation, standard error
        """
        mean_energy = energies.mean()
        var_energies = energies.var()
        stderr_energies = energies.std() / np.sqrt(len(energies))
        return mean_energy, var_energies, stderr_energies

    def _update_bqm(self, old_bqm, beta_nudge):
        linear = {u: beta_nudge * bias for u, bias in old_bqm.linear.items()}
        quadratic = {(u,v): beta_nudge * bias for (u,v), bias in old_bqm.quadratic.items()}
        return dimod.BQM(linear, quadratic, offset=0, vartype=dimod.SPIN)

    def estimate_ml(self, samples, learning_rate=0.25, **kwargs):
        """
        Maximum likelihood estimate by gradient descent

        Args:
            samples: SampleSet
            learning_rate: float
            **kwargs: args to pass to other functions
                Eg., num_reads for OrangSampler

        Returns:
            Array of betas
        """
        orang_sampler = OrangSampler()

        if kwargs.get('elimination_order') is not None:
            elim_order = kwargs.pop('elimination_order')
        else:
            elim_order = None

        beta_ml_estimate = {}
        for tile_num, tile in list(self.tiles.items())[:3]:
            qpu_energies = tile.qpu_tile_energy(samples, self.tiling_composite)
            qpu_mean_en, qpu_std_en, qpu_stderr_en = self.compute_energy_stats(qpu_energies)

            print('* ', qpu_energies)

            if elim_order is not None:
                elim_vars_in_tile = [v for v in elim_order if v in tile.bqm.variables]
            else:
                elim_vars_in_tile = None

            betas = [1]
            # TODO: replace with tolerance-specifc while loop
            for i in range(6):
                print("BETA here", betas[-1])
                new_bqm = self._update_bqm(tile.bqm, betas[-1])

                resp = orang_sampler.sample(new_bqm, elimination_order=elim_vars_in_tile, **kwargs)

                print(resp.record.energy)

                mean_en, var_en, stderr_en = self.compute_energy_stats(resp.record.energy)

                print("qpu_mean - mean = {}".format(qpu_mean_en - mean_en))

                beta = betas[-1] - 0.01 * np.random.normal()  # (learning_rate / max(1, var_en)) * (qpu_mean_en - mean_en)
                betas.append(beta)

                print("BETA", betas[-1])
                print(len(betas))

            print()

            beta_ml_estimate[tile_num] = betas[-1]

        return betas

    def fill_in_active_qubits_df(self, sampleset):
        """
        A tiling of the QPU does not cover all qubits. Since it's a heuristic algorithm we can use this method to
        fill unused qubits with zeros so that we always analyze the same Chimera graph.

        Args:
            sampleset: SampleSet object from QPU with N < num_qubits on QPU
            convert_to_df: boolean, toggle whether to return num_samples x num_qubits array or DataFrame with energies
            number of occurrences of each sample.

        Returns:
            DataFrame
        """
        samples = self.fill_in_active_qubits(sampleset)

        df_ = pd.DataFrame.from_records([samples, sampleset.record.energy, sampleset.record.num_occurrences]).T
        df_.columns = ['sample', 'energy', 'num_occurrences']
        return df_

    def fill_in_active_qubits(self, sampleset):
        """
        A tiling of the QPU does not cover all qubits. Since it's a heuristic algorithm we can use this method to
        fill unused qubits with zeros so that we always analyze the same Chimera graph.

        Args:
            sampleset: SampleSet object from QPU with N < num_qubits on QPU

        Returns:
            ndarray or DataFrame
        """
        num_qubits = self.tiling_composite.properties['child_properties']['num_qubits']

        active_qubits = []
        for emb in self.tiling_composite.embeddings:
            active_qubits += [next(iter(u)) for u in emb.values()]

        samples = np.zeros((sampleset.record.sample.shape[0], num_qubits), dtype=np.int8)
        for i, smp in enumerate(sampleset.record.sample):
            samples[i, active_qubits] = smp

        return samples

    # def get_tile_energy(self, tile_id, sampleset):
    #     tile = self.tiles[tile_id]
    #     new_bqm = tile.bqm.relabel_variables({j: i for i, j in enumerate(tile.bqm.variables)}, inplace=False)
    #     var2idx = [i for i, v in enumerate(self.tiling_composite.child.properties['qubits'])
    #                if v in tile.bqm.variables]
    #
    #     return new_bqm.energies(sampleset.record.sample[:, list(tile.bqm.variables)])

    def compute_tile_energy(self, tile, sample_df):
        new_bqm = tile.bqm.relabel_variables({j: i for i, j in enumerate(tile.bqm.variables)}, inplace=False)
        for sample in sample_df['sample']:
            print(new_bqm.energy(sample))


if __name__ == "__main__":

    R = Tiling(sub_m=2, sub_n=2)
    bqm_ = R.generate_ran_r_tiles()

    # from bqmtools import graphtools
    # graphtools.draw_bqm_on_chimera(bqm_)

    res = R.sample_tiling(num_reads=100)
    df = R.fill_in_active_qubits(res)
