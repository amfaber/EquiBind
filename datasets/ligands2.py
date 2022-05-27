from torch.utils.data import Dataset
from commons.utils import pmap_multi
from commons.process_mols import get_geometry_graph, get_lig_graph_revised
from dgl import batch
from commons.geometry_utils import random_rotation_translation
from rdkit.Chem import SDMolSupplier, SanitizeMol, SanitizeFlags, PropertyMol


def rand_trans_rot_lig_graph(lig_graph, use_rdkit_coords):
    rot_T, rot_b = random_rotation_translation(translation_distance=5)
    if (use_rdkit_coords):
        lig_coords_to_move = lig_graph.ndata['new_x']
    else:
        lig_coords_to_move = lig_graph.ndata['x']
    mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
    input_coords = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
    lig_graph.ndata['new_x'] = input_coords


class Ligands(Dataset):
    def __init__(self, ligpath, rec_graph, args, lazy = False, slice = None, skips = None):
        self.ligpath = ligpath
        self.rec_graph = rec_graph
        self.args = args
        self.dp = args.dataset_params
        self.use_rdkit_coords = args.use_rdkit_coords
        self.device = args.device
        self.lazy = lazy
        self.supplier = SDMolSupplier(ligpath, sanitize = False, removeHs=False)
        self.failed_ligs = []
        self.slice = None
        self.true_idx = []
        self.skips = skips if skips is not None else set()

        if not lazy:
            if slice is not None:
                slice = (slice[0] if slice[0] >= 0 else len(self.supplier)+slice[0], slice[1] if slice[1] >= 0 else len(self.supplier)+slice[1])
                self.slice = tuple(slice)
            else:
                self.slice = 0, len(self.supplier)
            self.ligs = []
            for i in range(*self.slice):
                if i in self.skips:
                    continue
                lig = self.supplier[i]
                if lig is not None:
                    sanitize_succeded = (SanitizeMol(lig, catchErrors = True) is SanitizeFlags.SANITIZE_NONE)
                    if sanitize_succeded:
                        self.ligs.append(PropertyMol.PropertyMol(lig))
                        self.true_idx.append(i)
                    else:
                        self.failed_ligs.append((i, lig.GetProp("_Name")))
                else:
                    self.failed_ligs.append((i, None))

    def __len__(self):
        if self.lazy:
            return len(self.supplier)
        else:
            return len(self.ligs)

    def __getitem__(self, idx):
        if self.lazy:
            true_index = idx
            if true_index in self.skips:
                return true_index, "Skipped"
            lig = self.supplier[true_index]
            if lig is None:
                self.failed_ligs.append((true_index, None))
                return true_index, None
            else:
                sanitize_succeded = (SanitizeMol(lig, catchErrors = True) is SanitizeFlags.SANITIZE_NONE)
                if sanitize_succeded:
                    lig = PropertyMol.PropertyMol(lig)
                else:
                    self.failed_ligs.append((true_index, None))
                    return true_index, None
        elif not self.lazy:
            lig = self.ligs[idx]
            true_index = self.true_idx[idx]
        
        lig_graph = get_lig_graph_revised(lig, lig.GetProp('_Name'), max_neighbors=self.dp['lig_max_neighbors'],
                                          use_rdkit_coords=self.use_rdkit_coords, radius=self.dp['lig_graph_radius'])

        geometry_graph = get_geometry_graph(lig) if self.dp['geometry_regularization'] else None
        if lig_graph is None:
            self.failed_ligs.append((true_index, lig.GetProp("_Name")))
            return true_index, lig.GetProp("_Name")
        # lig_graph = lig_graph.to(self.args.device)
        # geometry_graph = geometry_graph.to(self.args.device)


        lig_graph.ndata["new_x"] = lig_graph.ndata["x"]
        return lig, lig_graph.ndata["new_x"], lig_graph, self.rec_graph, geometry_graph, true_index
    
    @staticmethod
    def collate(_batch):
        sample_succeeded = lambda sample: not isinstance(sample[0], int)
        sample_failed = lambda sample: isinstance(sample[0], int)
        clean_batch = tuple(filter(sample_succeeded, _batch))
        failed_in_batch = tuple(filter(sample_failed, _batch))
        if len(clean_batch) == 0:
            return None, None, None, None, None, None, failed_in_batch
        ligs, lig_coords, lig_graphs, rec_graphs, geometry_graphs, true_indices = map(list, zip(*clean_batch))
        output = (
            ligs,
            lig_coords,
            batch(lig_graphs),
            batch(rec_graphs),
            batch(geometry_graphs) if geometry_graphs[0] is not None else None,
            true_indices,
            failed_in_batch
        )
        return output
