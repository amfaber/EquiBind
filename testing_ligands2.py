import equibind_inference 
if __name__ == "__main__":
    equibind_inference.main(["-l", "/home/qzj517/POR-DD/data/raw_data/cyp_screen/test_3D_opt_1216.sdf",
    "-r", "/home/qzj517/POR-DD/data/raw_data/cyp_screen/protein.pdb", "-o", "/home/qzj517/POR-DD/data/equibind_processed/testing", 
    "--n_workers_data_load", "0", "--batch_size", "32", "--no_skip", "--lazy_dataload"])