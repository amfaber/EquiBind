#!/usr/bin/env python
import argparse
import sys

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


# turn on for debugging C code like Segmentation Faults
import faulthandler

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
    
    if arglist is None:
        return p.parse_args()
    else:
        return p.parse_args(arglist)


def inference_from_files(args):
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    print(f"device = {device}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    # all_ligs_coords_corrected = []
    # all_intersection_losses = []
    # all_intersection_losses_untuned = []
    # all_ligs_coords_pred_untuned = []
    # all_ligs_coords = []
    # all_ligs_keypts = []
    # all_recs_keypts = []
    # all_names = []
    dp = args.dataset_params
    use_rdkit_coords = args.use_rdkit_coords if args.use_rdkit_coords != None else args.dataset_params['use_rdkit_coords']

    model = EquiBind(device = device, lig_input_edge_feats_dim = 15, rec_input_edge_feats_dim = 27, **args.model_parameters)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    

    def run_equibind(lig, lig_graph, rec_graph, model):
        name = lig.GetProp("_Name")

        if 'geometry_regularization' in dp and dp['geometry_regularization']:
            geometry_graph = get_geometry_graph(lig)
        elif 'geometry_regularization_ring' in dp and dp['geometry_regularization_ring']:
            geometry_graph = get_geometry_graph_ring(lig)
        else:
            geometry_graph = None

        # Randomly rotate and translate the ligand.
        rot_T, rot_b = random_rotation_translation(translation_distance=5)
        if (use_rdkit_coords):
            lig_coords_to_move = lig_graph.ndata['new_x']
        else:
            lig_coords_to_move = lig_graph.ndata['x']
        mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
        input_coords = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
        lig_graph.ndata['new_x'] = input_coords
        

        with torch.no_grad():
            geometry_graph = geometry_graph.to(device) if geometry_graph != None else None
            ligs_coords_pred_untuned, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss = model(
                lig_graph.to(device), rec_graph.to(device), geometry_graph,
                complex_names=[name])

            if args.run_corrections:
                prediction = ligs_coords_pred_untuned[0].detach().cpu()
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
                # all_ligs_coords_corrected.append(coords_pred_optimized)
                for i in range(optimized_mol.GetNumAtoms()):
                    x, y, z = coords_pred_optimized[i]
                    optimized_conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
                return optimized_mol


    def PDBBind_style():
        def load_lig_and_rec(idx, name):
            print(f'\nProcessing {name}: complex {idx + 1} of {len(names)}')
            file_names = os.listdir(os.path.join(args.inference_path, name))
            rec_name = [i for i in file_names if 'rec.pdb' in i or 'protein' in i][0]
            lig_names = [i for i in file_names if 'ligand' in i]
            rec_path = os.path.join(args.inference_path, name, rec_name)
            for lig_name in lig_names:
                if not os.path.exists(os.path.join(args.inference_path, name, lig_name)):
                    raise ValueError(f'Path does not exist: {os.path.join(args.inference_path, name, lig_name)}')
                print(f'Trying to load {os.path.join(args.inference_path, name, lig_name)}')
                lig = read_molecule(os.path.join(args.inference_path, name, lig_name), sanitize=True)
                if lig != None:  # read mol2 file if sdf file cannot be sanitized
                    used_lig = os.path.join(args.inference_path, name, lig_name)
                    break
            if lig_names == []: raise ValueError(f'No ligand files found. The ligand file has to contain \'ligand\'.')
            if lig == None: raise ValueError(f'None of the ligand files could be read: {lig_names}')
            print(f'Docking the receptor {os.path.join(args.inference_path, name, rec_name)}\nTo the ligand {used_lig}')
            
            rec, rec_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(rec_path)
            rec_graph = get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords,
                                    use_rec_atoms=dp['use_rec_atoms'], rec_radius=dp['rec_graph_radius'],
                                    surface_max_neighbors=dp['surface_max_neighbors'],
                                    surface_graph_cutoff=dp['surface_graph_cutoff'],
                                    surface_mesh_cutoff=dp['surface_mesh_cutoff'],
                                    c_alpha_max_neighbors=dp['c_alpha_max_neighbors'])
            lig_graph = get_lig_graph_revised(lig, name, max_neighbors=dp['lig_max_neighbors'],
                                            use_rdkit_coords=use_rdkit_coords, radius=dp['lig_graph_radius'])
            return lig, lig_graph, rec_graph
        
        def save_to_output(optimized_mol, name):
            os.makedirs(f'{args.output_directory}/{name}', exist_ok=True)
            block_optimized = Chem.MolToMolBlock(optimized_mol)
            print(f'Writing prediction to {args.output_directory}/{name}/lig_equibind_corrected.sdf')
            with open(f'{args.output_directory}/{name}/lig_equibind_corrected.sdf', "w") as newfile:
                    newfile.write(block_optimized)

        names = os.listdir(args.inference_path) if args.inference_path != None else tqdm(read_strings_from_txt('data/timesplit_test'))

        check_skip = False
        if os.path.exists(args.output_directory) and args.skip_in_output:
            check_skip = True
            to_skip = os.listdir(args.output_directory)
            to_skip = [name.replace("_failed", "") for name in to_skip]
        
        for idx, name in enumerate(names):
            if check_skip:
                if name in to_skip:
                    print(f"Skipping {name}")
                    continue
            try:
                lig, lig_graph, rec_graph = load_lig_and_rec(idx, name)
                optimized_mol = run_equibind(lig, lig_graph, rec_graph, model)
                if args.output_directory:
                    save_to_output(optimized_mol, name)
                # all_names.append(name)
            except Exception as e:
                print(f"Something failed on {name}")
                print(e)
                if not os.path.exists(f'{args.output_directory}/{name}_failed'):
                    os.makedirs(f'{args.output_directory}/{name}_failed')

    def predict_for_lig(lig, rec_graph, model):
        name = lig.GetProp("_Name")
        try:
            lig_graph = get_lig_graph_revised(lig, name, max_neighbors=dp['lig_max_neighbors'],
                                              use_rdkit_coords=use_rdkit_coords, radius=dp['lig_graph_radius'])
            opt_mol = run_equibind(lig, lig_graph, rec_graph, model)
            return name, opt_mol
        except Exception:
            return name, None

    
    def screening_style():
        assert args.output_directory, "An output directory should be specified"
        ligs, names = read_molecules_from_sdf(args.ligands_sdf, sanitize = True, return_names = True)
        n_ligs = len(ligs)
        rec_path = args.rec_pdb
        rec, rec_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(rec_path)
        rec_graph = get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords,
                                    use_rec_atoms=dp['use_rec_atoms'], rec_radius=dp['rec_graph_radius'],
                                    surface_max_neighbors=dp['surface_max_neighbors'],
                                    surface_graph_cutoff=dp['surface_graph_cutoff'],
                                    surface_mesh_cutoff=dp['surface_mesh_cutoff'],
                                    c_alpha_max_neighbors=dp['c_alpha_max_neighbors'])
        

        os.makedirs(args.output_directory, exist_ok = True)
        full_output_path = os.path.join(args.output_directory, "output.sdf")
        full_failed_path = os.path.join(args.output_directory, "failed.txt")
        full_success_path = os.path.join(args.output_directory, "success.txt")
        if os.path.exists(full_output_path) and args.skip_in_output:
            supplier = Chem.SDMolSupplier(full_output_path)
            to_skip = [mol.GetProp("_Name") for mol in supplier]
            skipping = True
        else:
            skipping = False
        
        with open(full_output_path, "a") as file, open(full_failed_path, "a") as failed_file, open(
            full_success_path, "a") as success_file:
            with Chem.SDWriter(file) as writer:
                for i, (lig, name) in enumerate(zip(ligs, names)):
                    if skipping:
                        if name in to_skip:
                            print(f"({i+1}/{n_ligs}) skipped {name}")
                            continue
                    name, opt_mol = predict_for_lig(lig, rec_graph, model)
                    if not opt_mol is None:
                        writer.write(opt_mol)
                        success_file.write(f"{i} {name}\n")
                        print(f"({i+1}/{n_ligs}) Processed and wrote {name} to output.sdf")
                    else:
                        print(f"({i+1}/{n_ligs}) Failed on {name}, printing to failed.txt")
                        failed_file.write(f"{i} {name}\n")
                    # try:
                    #     lig_graph = get_lig_graph_revised(lig, name, max_neighbors=dp['lig_max_neighbors'],
                    #                                     use_rdkit_coords=use_rdkit_coords, radius=dp['lig_graph_radius'])
                    #     optimized_mol = run_equibind(lig, lig_graph, rec_graph, model)
                    #     writer.write(optimized_mol)
                    #     success_file.write(f"{i} {name}\n")
                    # except Exception as e:
                    #     print(f"({i+1}/{n_ligs}) Failed on {name}, printing to failed.txt")
                    #     failed_file.write(f"{i} {name}\n")
                    


    
    if args.PDBBind:
        PDBBind_style()
    else:
        screening_style()



    # path = os.path.join(os.path.dirname(args.checkpoint), f'predictions_RDKit{use_rdkit_coords}.pt')
    # print(f'Saving predictions to {path}')
    # results = {'corrected_predictions': all_ligs_coords_corrected, 'initial_predictions': all_ligs_coords_pred_untuned,
    #            'targets': all_ligs_coords, 'lig_keypts': all_ligs_keypts, 'rec_keypts': all_recs_keypts,
    #            'names': all_names, 'intersection_losses_untuned': all_intersection_losses_untuned,
    #            'intersection_losses': all_intersection_losses}
    # torch.save(results, path)


def get_default_args(args):
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
    #for run_dir in args.run_dirs:
    args.checkpoint = os.path.join(os.path.dirname(__file__), f'runs/{run_dir}/best_checkpoint.pt')
    config_dict['checkpoint'] = f'runs/{run_dir}/best_checkpoint.pt'
    # overwrite args with args from checkpoint except for the args that were contained in the config file
    arg_dict = args.__dict__
    with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
        checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
    for key, value in checkpoint_dict.items():
        if key not in config_dict.keys():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
    args.model_parameters['noise_initial'] = 0
    return args

if __name__ == '__main__':
    args = parse_arguments()
    
    args = get_default_args(args)
    
    inference_from_files(args)
