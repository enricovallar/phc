#!/bin/bash
#BSUB -J BO_A2_E1_h_0.30
#BSUB -q hpc
#BSUB -n 30
#BSUB -W 24:00
#BSUB -R "rusage[mem=1GB]"
#BSUB -R "span[ptile=7]"
#BSUB -oo /work3/enrva/phc_nzi_data/MPB_data/C6v_hBN_slab_optimization/BO_A2_E1_h_0.30/BO_A2_E1_h_0.30.out
#BSUB -eo /work3/enrva/phc_nzi_data/MPB_data/C6v_hBN_slab_optimization/BO_A2_E1_h_0.30/BO_A2_E1_h_0.30.err
module purge
source /zhome/2f/7/202918/miniconda3/etc/profile.d/conda.sh
conda activate mpb-nzi-env
mpirun -np 30 python /zhome/2f/7/202918/phc_nzi/src/phc_nzi/mpi_bayesian_optimization.py --run_opt --param_names="r1,r2" --simulation_name="BO_A2_E1_h_0.30" --directory="/work3/enrva/phc_nzi_data/MPB_data/C6v_hBN_slab_optimization/BO_A2_E1_h_0.30" --maxiter=3 --polarization="zeven" --batch_size=30 --objective_mode="log" --strategy="cl_min" --target_cost=0.0025 --target_irreps A_2 E_1 E_1 --irrep_occurrences 1 4 4 --symmetry_group C6v --height_slab=0.3 --bo_options='{"dimensions": [[0.15, 0.25], [0.05, 0.1]], "acq_func": "LCB", "acq_func_kwargs": {"kappa": 3.5}, "n_initial_points": 60, "initial_point_generator": "sobol", "random_state": 42}' --fixed_params='{"x_dist": 0.25, "h": 0.3}'
