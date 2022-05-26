#!/usr/bin/env python
import argparse
import sys
from functools import partial

from copy import copy, deepcopy

import os

from dgl import load_graphs

from rdkit import Chem
from rdkit.Chem import RemoveHs
from rdkit.Geometry import Point3D
from tqdm import tqdm

from commons.geometry_utils import rigid_transform_Kabsch_3D, get_torsions, get_dihedral_vonMises, apply_changes
from commons.logger import Logger
from commons.process_mols import read_molecule, get_receptor, get_lig_graph_revised, \
    get_rec_graph, get_receptor_atom_subgraph, get_geometry_graph, get_geometry_graph_ring, \
    get_receptor_inference, read_molecules_from_sdf

#from train import load_model

from datasets.pdbbind import PDBBind

from commons.utils import seed_all, read_strings_from_txt

import yaml

from datasets.custom_collate import *  # do not remove
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove
from torch.utils.data import DataLoader


# turn on for debugging C code like Segmentation Faults
import faulthandler
from datasets import ligands

faulthandler.enable()

from models.equibind import EquiBind

def parse_arguments(arglist = None):
    p = argparse.ArgumentParser()    
    p.add_argument('--config', type=argparse.FileType(mode='r'), default=None)
    p.add_argument('--checkpoint', type=str, help='path to .pt file in a checkpoint directory')
    p.add_argument('-o', '--output_directory', type=str, default=None, help='path where to put the predicted results')
    p.add_argument('--run_dirs', type=list, default=["flexible_self_docking"], help='path directory with saved runs')
    p.add_argument('--fine_tune_dirs', type=list, default=[], help='path directory with saved finetuning runs')
    p.add_argument('--inference_path', type=str, help='path to some pdb files for which you want to run inference')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset_params', type=dict, default={},
                   help='parameters with keywords of the dataset')
    p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
    p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
    p.add_argument('--seed', type=int, default=1, help='seed for reproducibility')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=1, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--clip_grad', type=float, default=None, help='clip gradients if magnitude is greater')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100,
                   help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='loss', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--num_epochs_local_only', type=int, default=1,
                   help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='MPNN', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--trainer', type=str, default='binding', help='')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--check_se3_invariance', type=bool, default=False, help='check it instead of generating files')
    p.add_argument('--num_confs', type=int, default=1, help='num_confs if using rdkit conformers')
    p.add_argument('--use_rdkit_coords', action="store_true", help='override the rkdit usage behavior of the used model')
    p.add_argument('--no_skip', dest = "skip_in_output", action = "store_false", help = 'skip input files that already have corresponding folders in the output directory. Used to resume a large interrupted computation')
    p.add_argument('--PDBBind', action = "store_true", help = "Toggles whether or not to process in PDBBind mode or screening mode")
    p.add_argument("--no_run_corrections", dest = "run_corrections", action = "store_false", help = "possibility of turning off running fast point cloud ligand fitting")
    #p.add_argument("--no_use_rdkit_coords", action = "store_false", help = "Turn off rdkit coordinate randomization")
    p.add_argument("-l", "--ligands_sdf", type=str, help = "A single sdf file containing all ligands to be screened when running in screening mode")
    p.add_argument("-r", "--rec_pdb", type = str, help = "The receptor to dock the ligands in --ligands_sdf against")
    p.add_argument("--n_workers_data_load", type = int, default = 4, help = "The number of cores used for loading the ligands and generating the graphs used as input to the model")
    p.add_argument("--mess_with_seed", action = "store_true")
    cmdline_parser = deepcopy(p)
    args = p.parse_args(arglist)
    clear_defaults = {key: argparse.SUPPRESS for key in args.__dict__}
    cmdline_parser.set_defaults(**clear_defaults)
    cmdline_parser._defaults = {}
    cmdline_args = cmdline_parser.parse_args(arglist)

    return p.parse_args(arglist), set(cmdline_args.__dict__.keys())

def run_corrections(lig, lig_coord, ligs_coords_pred_untuned):
    input_coords = lig_coord.detach().cpu()
    prediction = ligs_coords_pred_untuned.detach().cpu()
    lig_input = deepcopy(lig)
    conf = lig_input.GetConformer()
    for i in range(lig_input.GetNumAtoms()):
        x, y, z = input_coords.numpy()[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    lig_equibind = deepcopy(lig)
    conf = lig_equibind.GetConformer()
    for i in range(lig_equibind.GetNumAtoms()):
        x, y, z = prediction.numpy()[i]
        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

    coords_pred = lig_equibind.GetConformer().GetPositions()

    Z_pt_cloud = coords_pred
    rotable_bonds = get_torsions([lig_input])
    new_dihedrals = np.zeros(len(rotable_bonds))
    for idx, r in enumerate(rotable_bonds):
        new_dihedrals[idx] = get_dihedral_vonMises(lig_input, lig_input.GetConformer(), r, Z_pt_cloud)
    optimized_mol = apply_changes(lig_input, new_dihedrals, rotable_bonds)
    optimized_conf = optimized_mol.GetConformer()
    coords_pred_optimized = optimized_conf.GetPositions()
    R, t = rigid_transform_Kabsch_3D(coords_pred_optimized.T, coords_pred.T)
    coords_pred_optimized = (R @ (coords_pred_optimized).T).T + t.squeeze()
    for i in range(optimized_mol.GetNumAtoms()):
        x, y, z = coords_pred_optimized[i]
        optimized_conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
    return optimized_mol

def load_statics(args):
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"device = {device}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    dp = args.dataset_params

    model = EquiBind(device = device, lig_input_edge_feats_dim = 15, rec_input_edge_feats_dim = 27, **args.model_parameters)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    ligs = read_molecules_from_sdf(args.ligands_sdf, sanitize = True)
    success_path = os.path.join(args.output_directory, "success.txt")
    failed_path = os.path.join(args.output_directory, "failed.txt")
    if os.path.exists(success_path) and os.path.exists(failed_path) and args.skip_in_output:
        with open(success_path) as successes, open(failed_path) as failures:
            previous_work = successes.read().splitlines()
            previous_work += failures.read().splitlines()
        print(f"Skipping {len(previous_work)} ligands in success.txt / failed.txt")
        previous_work = set(previous_work)
        ligs = [lig for lig in ligs if lig.GetProp('_Name') not in previous_work]
        print(f"{len(ligs)} ligands remain")

    rec_path = args.rec_pdb
    rec, rec_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(rec_path)
    rec_graph = get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords,
                                use_rec_atoms=dp['use_rec_atoms'], rec_radius=dp['rec_graph_radius'],
                                surface_max_neighbors=dp['surface_max_neighbors'],
                                surface_graph_cutoff=dp['surface_graph_cutoff'],
                                surface_mesh_cutoff=dp['surface_mesh_cutoff'],
                                c_alpha_max_neighbors=dp['c_alpha_max_neighbors'])
    rec_graph = rec_graph.to(device)

    return ligs, rec_graph, model

def get_default_args(args, cmdline_args):
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}
    
    run_dir=args.run_dirs[0]
    args.checkpoint = os.path.join(os.path.dirname(__file__), f'runs/{run_dir}/best_checkpoint.pt')
    config_dict['checkpoint'] = f'runs/{run_dir}/best_checkpoint.pt'
    # overwrite args with args from checkpoint except for the args that were contained in the config file
    arg_dict = args.__dict__
    with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
        checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
    for key, value in checkpoint_dict.items():
        if (key not in config_dict.keys()) and (key not in cmdline_args):
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    args.model_parameters['noise_initial'] = 0
    return args

def _run(batch, model):
    ligs, lig_coords, lig_graphs, rec_graphs, geometry_graphs = batch
    failsafe = lig_graphs.ndata['feat']
    try:
        predictions = model(lig_graphs, rec_graphs, geometry_graphs)[0]
        # print(f"Succeeded for {[mol.GetProp('_Name') for mol in ligs]}")
        out_ligs = ligs
        out_lig_coords = lig_coords
        failures = []
    except AssertionError:
        lig_graphs.ndata['feat'] = failsafe
        lig_graphs, rec_graphs, geometry_graphs = (dgl.unbatch(lig_graphs),
        dgl.unbatch(rec_graphs), dgl.unbatch(geometry_graphs))
        predictions = []
        out_ligs = []
        out_lig_coords = []
        failures = []
        for lig, lig_coord, lig_graph, rec_graph, geometry_graph in zip(ligs, lig_coords, lig_graphs, rec_graphs, geometry_graphs):
            try:
                output = model(lig_graph, rec_graph, geometry_graph)
                # print(f"Succeeded for {lig.GetProp('_Name')}")
            except AssertionError as e:
                failures.append(lig)
                print(f"Failed for {lig.GetProp('_Name')}")
            else:
                out_ligs.append(lig)
                out_lig_coords.append(lig_coord)
                predictions.append(output[0][0])
    assert len(predictions) == len(out_ligs)
    return out_ligs, out_lig_coords, predictions, failures

def io(dataloader, model, args):
    
    full_output_path = os.path.join(args.output_directory, "output.sdf")
    full_failed_path = os.path.join(args.output_directory, "failed.txt")
    full_success_path = os.path.join(args.output_directory, "success.txt")

    w_or_a = "a" if args.skip_in_output else "w"
    with torch.no_grad(), open(full_output_path, w_or_a) as file, open(
        full_failed_path, w_or_a) as failed_file, open(full_success_path, w_or_a) as success_file:
        with Chem.SDWriter(file) as writer:
            i = 0
            total_ligs = len(dataloader.dataset)
            for batch in dataloader:
                i += args.batch_size
                print(f"Entering batch ending in index {min(i,total_ligs)}/{len(dataloader.dataset)}")
                ligs, lig_coords, lig_graphs, rec_graphs, geometry_graphs = batch
                out_ligs, out_lig_coords, predictions, failures = _run(batch, model)
                opt_mols = [run_corrections(lig, lig_coord, prediction) for lig, lig_coord, prediction in zip(out_ligs, out_lig_coords, predictions)]
                for mol in opt_mols:
                    writer.write(mol)
                    success_file.write(mol.GetProp('_Name'))
                    success_file.write("\n")
                    # print(f"written {mol.GetProp('_Name')} to output")
                for failure in failures:
                    failed_file.write(failure.GetProp('_Name'))
                    failed_file.write("\n")

def main(arglist = None):
    args, cmdline_args = parse_arguments(arglist)
    
    args = get_default_args(args, cmdline_args)
    assert args.output_directory, "An output directory should be specified"
    assert args.ligands_sdf, "No ligand sdf specified"
    assert args.rec_pdb, "No protein specified"
    seed_all(args.seed)
    if args.mess_with_seed:
        torch.rand(1)

    os.makedirs(args.output_directory, exist_ok = True)
    
    ligs, rec_graph, model = load_statics(args)
    lig_data = ligands.Ligands(ligs, rec_graph, args, n_jobs=args.n_workers_data_load)

    full_failed_path = os.path.join(args.output_directory, "failed.txt")
    with open(full_failed_path, "a" if args.skip_in_output else "w") as failed_file:
        for lig in lig_data.failed_ligs:
            failed_file.write(lig)
            failed_file.write("\n")
        
    lig_loader = DataLoader(lig_data, batch_size = args.batch_size, collate_fn = lig_data.collate(rec_batch = True))

    io(lig_loader, model, args)

if __name__ == '__main__':

    main()