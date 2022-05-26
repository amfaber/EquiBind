#!/usr/bin/env python
import argparse
import os
from os.path import join
import equibind_inference
import rdkit.Chem as Chem
from shutil import copy2

def parse(arglist = None):
    p = argparse.ArgumentParser()
    p.add_argument("--trad", action="store_true")
    p.add_argument("--no_new", dest = "new", action="store_false")
    p.add_argument("--no_gnina", dest = "gnina", action="store_false")
    p.add_argument("--gnina", dest = "gnina", action="store_true")
    p.add_argument("--force_gnina", dest = "force", action="store_true")
    p.add_argument("--seed", type=int, default = 1)
    p.add_argument("--use_rdkit_coords", action="store_true")
    p.add_argument("--mess_with_seed", action="store_true")

    return p.parse_args(arglist)

def run_traditional(args):
    cmd = f"python inference.py --config=configs_clean/unit_test.yml --seed {args.seed}"
    if args.use_rdkit_coords:
        cmd += " --use_rdkit_coords"
    else:
        cmd += " --no_use_rdkit_coords"
    os.system(cmd)
    results_path = "data/equibind_inference_test/traditional_output"
    folders = next(os.walk(results_path))[1]
    mols = []
    for folder in folders:
        file_path = join(results_path, folder, "lig_equibind_corrected.sdf")
        mols.append(next(Chem.SDMolSupplier(file_path, removeHs = False)))
    
    correct_order = Chem.SDMolSupplier("data/equibind_inference_test/first_10_ligs.sdf")
    correct_order = {mol.GetProp("_Name"): i for i, mol in enumerate(correct_order)}
    key = lambda mol: correct_order[mol.GetProp('_Name')]
    mols.sort(key = key)
    
    with Chem.SDWriter("data/equibind_inference_test/ultimate_outputs/old.sdf") as all:
        for mol in mols:
            all.write(mol)

def run_new(args):
    _args = ["-o", "data/equibind_inference_test/new",
    "-l", "data/equibind_inference_test/first_10_ligs.sdf",
    "-r", "../data/raw_data/cyp_screen/protein.pdb",
    "--no_skip",
    "--n_workers_data_load", "1",
    "--seed", f"{args.seed}",
    ]
    if args.use_rdkit_coords:
        _args += ["--use_rdkit_coords"]

    
    equibind_inference.main(_args)
    copy2("data/equibind_inference_test/new/output.sdf", "data/equibind_inference_test/ultimate_outputs/new.sdf")
    if args.mess_with_seed:
        equibind_inference.main(_args + ["--mess_with_seed"])
        copy2("data/equibind_inference_test/new/output.sdf", "data/equibind_inference_test/ultimate_outputs/new2.sdf")
    



def gnina(old, new, force):
    cmd = "gnina -r ../data/raw_data/cyp_screen/protein.pdb -l data/equibind_inference_test/ultimate_outputs/old.sdf --minimize > data/equibind_inference_test/gnina/old.txt"
    cmd2 = "gnina -r ../data/raw_data/cyp_screen/protein.pdb -l data/equibind_inference_test/ultimate_outputs/new.sdf --minimize > data/equibind_inference_test/gnina/new.txt"
    if old or force:
        os.system(cmd)
    if new or force:
        os.system(cmd2)

if __name__ == "__main__":
    args = parse()
    if args.trad:
        run_traditional(args)
    if args.new:
        run_new(args)
    if args.gnina or args.force:
        gnina(args.trad, args.new, args.force)
