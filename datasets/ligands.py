from torch.utils.data import Dataset
from commons.utils import pmap_multi
from commons.process_mols import get_geometry_graph, get_lig_graph_revised
from dgl import batch
from commons.geometry_utils import random_rotation_translation

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
    def __init__(self, ligs, rec_graph, args, n_jobs = 4, verbose = 1):
        self.rec_graph = rec_graph
        self.args = args
        self.n_jobs = n_jobs
        dp = args.dataset_params
        use_rdkit_coords = args.use_rdkit_coords
        device = args.device
        names = [lig.GetProp("_Name") for lig in ligs]
        kwargs = dict(max_neighbors=dp['lig_max_neighbors'],
                        use_rdkit_coords=use_rdkit_coords, radius=dp['lig_graph_radius'])
        # self.lig_graphs = [get_lig_graph_revised(lig, name, **kwargs) for lig, name in zip(ligs, names)]
        
        lig_graphs = pmap_multi(get_lig_graph_revised, zip(ligs, names), n_jobs=self.n_jobs, verbose = verbose, **kwargs)
        
        if 'geometry_regularization' in dp and dp['geometry_regularization']:
            geometry_graphs = pmap_multi(get_geometry_graph, zip(ligs), n_jobs = self.n_jobs, verbose = verbose)
        else:
            geometry_graphs = None
        
        to_remove = set()
        self.failed_ligs = []
        for i in range(len(lig_graphs)):
            if lig_graphs[i] is None:
                to_remove.add(i)
                self.failed_ligs.append(ligs[i].GetProp('_Name'))
                print(f"Graph generation failed for {ligs[i].GetProp('_Name')}")

        
        
        self.ligs = self._remove_indices(ligs, to_remove)
        self.lig_graphs = self._remove_indices(lig_graphs, to_remove)
        self.geometry_graphs = self._remove_indices(geometry_graphs, to_remove)

        [rand_trans_rot_lig_graph(lig_graph, use_rdkit_coords) for lig_graph in self.lig_graphs]
        
        self.lig_graphs = [graph.to(device) for graph in self.lig_graphs]
        self.geometry_graphs = [graph.to(device) for graph in self.geometry_graphs]
        # self.lig_graphs = pmap_multi(rand_trans_rot_lig_graph, self.lig_graphs, n_jobs=self.n_jobs, use_rdkit_coords = use_rdkit_coords)

    def _remove_indices(self, List, indices):
        return [List[i] for i in range(len(List)) if i not in indices]

    def __len__(self):
        return len(self.ligs)

    def __getitem__(self, idx):
        return self.ligs[idx], self.lig_graphs[idx].ndata["new_x"], self.lig_graphs[idx], self.rec_graph, self.geometry_graphs[idx]
    
    def collate(self, rec_batch = True):

        def _collate(_batch):
            ligs, lig_coords, lig_graphs, rec_graphs, geometry_graphs = map(list, zip(*_batch))
            output = (
                ligs,
                lig_coords,
                batch(lig_graphs),
                batch(rec_graphs) if rec_batch else rec_graphs[0],
                batch(geometry_graphs) if geometry_graphs[0] is not None else None
            )
            return output
        
        return _collate