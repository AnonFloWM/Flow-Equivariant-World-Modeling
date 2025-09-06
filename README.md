# Flow Equivariant World Models

## Creating a Conda Environment

To set up the project environment, follow these steps:

**Create a Conda Environment**

Create a new conda env, activate, and then install pip requirements
```bash
conda create --name fernn-wm python=3.12 -y
conda activate fernn-wm
pip install -r requirements.txt
```

## Code Structure
The main training file for MNIST World is `train_dynamic.py`, the datasets are defined in `dynamic_mnist_world_dataset.py` and the models are defined in `models.py`. 


## Running the experiments from the paper

### MNIST World Dataset

#### FERNN V2
Dynamic:
```bash
python train_dynamic.py --model_type 'fernn' --v_range 2 --cell_type v_channels --run_name dynamic_fernn_v2_w_sme_load --split dynamic --run_len_gen
```
Static:
```bash
python train_dynamic.py --model_type 'fernn' --v_range 2 --cell_type v_channels --run_name static_fernn_v2_w_sme_load --split static --run_len_gen
```
Dynmaic Fully Observed
```bash
python train_dynamic.py --model_type 'fernn' --v_range 2 --cell_type v_channels --run_name dynamic_fixedFO_fernn_v2_w_sme_load --split dynamic_smallworld --run_len_gen --window_size 32 --world_size 32
```
Dynamic Fully Observed, no Self-Motion
```bash
python train_dynamic.py --model_type 'fernn' --v_range 2 --cell_type v_channels --run_name dynamic_fixedFO_no-em_fernn_v2_w_sme_load --split dynamic_smallworld_no_em --run_len_gen --window_size 32 --world_size 32
```

#### FERNN V2 no self motion equivariance
```bash
python train_dynamic.py --model_type 'fernn' --v_range 2 --cell_type v_channels --run_name dynamic_fernn_v2_no-sme_load --split dynamic --no_self_motion_equivariance --run_len_gen
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 2 --cell_type v_channels --run_name static_fernn_v2_no-sme_load --split static --no_self_motion_equivariance --run_len_gen
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 2 --cell_type v_channels --run_name dynamic_fixedFO_fernn_v2_no-sme_load --split dynamic_smallworld --no_self_motion_equivariance --run_len_gen --window_size 32 --world_size 32
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 2 --cell_type v_channels --run_name dynamic_fixedFO_no-em_fernn_v2_no-sme_load --split dynamic_smallworld_no_em --no_self_motion_equivariance --run_len_gen --window_size 32 --world_size 32
```

#### FERNN V0 (No velocity channels)
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type v_channels --run_name dynamic_fernn_v0_w_sme_load --split dynamic --run_len_gen
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type v_channels --run_name static_fernn_v0_w_sme_load --split static --run_len_gen
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type v_channels --run_name dynamic_fixedFO_fernn_v0_w_sme_load --split dynamic_smallworld --run_len_gen --window_size 32 --world_size 32
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type v_channels --run_name dynamic_fixedFO_no-em_fernn_v0_w_sme_load --split dynamic_smallworld_no_em --run_len_gen --window_size 32 --world_size 32
```

#### FERNN V0 (No velocity channels) no self motion equivariance
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type v_channels --run_name dynamic_fernn_v0_no-sme_load --split dynamic --no_self_motion_equivariance --run_len_gen
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type v_channels --run_name static_fernn_v0_no-sme_load --split static --no_self_motion_equivariance --run_len_gen
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type v_channels --run_name dynamic_fixedFO_fernn_v0_no-sme_load --split dynamic_smallworld --no_self_motion_equivariance --run_len_gen --window_size 32 --world_size 32
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type v_channels --run_name dynamic_fixedFO_no-em_fernn_v0_no-sme_load --split dynamic_smallworld_no_em --no_self_motion_equivariance --run_len_gen --window_size 32 --world_size 32
```

#### Action concat
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type action_concat --run_name dynamic_fernn_v0_action_concat_load --split dynamic --run_len_gen
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type action_concat --run_name static_fernn_v0_action_concat_load --split static --run_len_gen
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type action_concat --run_name dynamic_fixedFO_fernn_v0_action_concat_load --split dynamic_smallworld --run_len_gen --window_size 32 --world_size 32
```
```bash
python train_dynamic.py --model_type 'fernn' --v_range 0 --cell_type action_concat --run_name dynamic_fixedFO_no-em_fernn_v0_action_concat_load --split dynamic_smallworld_no_em --run_len_gen --window_size 32 --world_size 32
```