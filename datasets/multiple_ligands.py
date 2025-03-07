from torch.utils.data import Dataset
from commons.process_mols import get_geometry_graph, get_lig_graph_revised, get_rdkit_coords
from dgl import batch
from rdkit.Chem import SDMolSupplier, SanitizeMol, SanitizeFlags, PropertyMol, SmilesMolSupplier, AddHs, MultithreadedSmilesMolSupplier, MultithreadedSDMolSupplier


def safe_get_name(lig):
    try:
        return lig.GetProp("_Name")
    except KeyError:
        return None

class Ligands(Dataset):
    def __init__(
            self, ligpath, rec_graph, args,
            skips = None, ext = None,
            rdkit_seed = None, lig_load_workers = 0,
            keeps = None,
            lazy = False
            ):
        self.ligpath = ligpath
        self.rec_graph = rec_graph
        self.args = args
        self.dp = args.dataset_params
        self.device = args.device
        self.rdkit_seed = rdkit_seed
        self.generate_conformer = args.use_rdkit_coords
        self.addH = args.addH
        
        ##Default argument handling
        self.skips = skips
        self.keeps = keeps
        self.lazy = lazy

        extensions_requiring_conformer_generation = ["smi"]

        if ext is None:
            try:
                ext = ligpath.split(".")[-1]
            except (AttributeError, KeyError):
                ext = "sdf"

        if self.addH is None:
            if ext == "smi":
                self.addH = True
            else:
                self.addH = False
        
        if self.generate_conformer is None:
            self.generate_conformer = ext in extensions_requiring_conformer_generation

        if lig_load_workers > 0 and not self.lazy:
            suppliers = {"sdf": MultithreadedSDMolSupplier, "smi": MultithreadedSmilesMolSupplier}
            supp_kwargs = {"sdf": dict(sanitize = False, removeHs =  False, numWriterThreads = lig_load_workers),
                            "smi": dict(sanitize = False, titleLine = False, numWriterThreads = lig_load_workers)}
            self.supplier = suppliers[ext](ligpath, **supp_kwargs[ext])
            print("start loading ligs")
            self.ligs = [(lig, self.supplier.GetLastRecordId()) for lig in self.supplier]
            self.ligs = sorted(self.ligs, key = lambda tup: tup[1])
            self.ligs = list(zip(*self.ligs))[0][:-1]
            print("finish loading ligs")
        else:
            suppliers = {"sdf": SDMolSupplier, "smi": SmilesMolSupplier}
            supp_kwargs = {"sdf": dict(sanitize = False, removeHs =  False),
                            "smi": dict(sanitize = False, titleLine = False)}
            self.supplier = suppliers[ext](ligpath, **supp_kwargs[ext])
            if not self.lazy:
                self.ligs = [lig for lig in self.supplier]
            else:
                self.ligs = None
        
        if self.ligs is not None:
            self._len = len(self.ligs)
        else:
            self._len = len(self.supplier)

    def _process(self, lig):
        if lig is None:
            return None, None
        if self.addH:
            sanitize_succeded = (SanitizeMol(lig, catchErrors = True) is SanitizeFlags.SANITIZE_NONE)
            if not sanitize_succeded:
                return None, safe_get_name(lig)
            lig = AddHs(lig)
        if self.generate_conformer:
            try:
                get_rdkit_coords(lig, self.rdkit_seed)
            except ValueError:
                return None, safe_get_name(lig)
        sanitize_succeded = (SanitizeMol(lig, catchErrors = True) is SanitizeFlags.SANITIZE_NONE)
        if self.args.lig_name is not None:
            lig.SetProp("_Name", lig.GetProp(self.args.lig_name))
        if sanitize_succeded:
            return lig, safe_get_name(lig)
        else:
            return None, safe_get_name(lig)

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        if self.ligs is not None:
            lig = self.ligs[idx]
        else:
            lig = self.supplier[idx]

        if self.skips is not None and idx in self.skips:
            return idx, "Skipped"
        
        if self.keeps is not None and idx not in self.keeps:
            return idx, "Skipped"
        
        lig, name = self._process(lig)
        if lig is None:
            return idx, name
        else:
            lig = PropertyMol.PropertyMol(lig)

        try:
            lig_graph = get_lig_graph_revised(lig, safe_get_name(lig), max_neighbors=self.dp['lig_max_neighbors'],
                                            use_rdkit_coords=False, radius=self.dp['lig_graph_radius'])
        except AssertionError:
            return idx, safe_get_name(lig)
        
        geometry_graph = get_geometry_graph(lig) if self.dp['geometry_regularization'] else None

        lig_graph.ndata["new_x"] = lig_graph.ndata["x"]
        return lig, lig_graph.ndata["new_x"], lig_graph, self.rec_graph, geometry_graph, idx
    
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
