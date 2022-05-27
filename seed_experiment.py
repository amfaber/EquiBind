import equibind_inference 
if __name__ == "__main__": 
    def make_args(seed):
        return ["-l", "/home/qzj517/POR-DD/data/raw_data/cyp_screen/test_3D_opt_1216.sdf",
    "-r", "/home/qzj517/POR-DD/data/raw_data/cyp_screen/protein.pdb", "-o",
    f"/home/qzj517/POR-DD/data/equibind_processed/seed_experiment/seed_{seed}", 
    "--n_workers_data_load", "4", "--batch_size", "32", "--no_skip", "--seed", f"{seed}"]

    for i in range(20):
        equibind_inference.main(make_args(i))