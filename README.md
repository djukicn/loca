# A Low-Shot Object Counting Network With Iterative Prototype Adaptation (ICCV 2023)

The official PyTorch implementation of the [ICCV 2023 paper LOCA](https://openaccess.thecvf.com/content/ICCV2023/papers/Dukic_A_Low-Shot_Object_Counting_Network_With_Iterative_Prototype_Adaptation_ICCV_2023_paper.pdf).

```
@InProceedings{Dukic_2023_ICCV,
    author    = {{\DJ}uki\'c, Nikola and Luke\v{z}i\v{c}, Alan and Zavrtanik, Vitjan and Kristan, Matej},
    title     = {A Low-Shot Object Counting Network With Iterative Prototype Adaptation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2023},
    pages     = {18872-18881}
}
```

## Setup

### Install the libraries

To run the code, install the following libraries: `PyTorch 1.11.0`, `Torchvision 0.12.0`, `scipy`, `numpy` and `PIL`.

### Download the dataset

Download the FSC147 dataset as instructed in its [official repository](https://github.com/cvlab-stonybrook/LearningToCountEverything). Make sure to
also download the `annotation_FSC147_384.json` and `Train_Test_Val_FSC_147.json` and place them alongside the image directory (`images_384_VarV2`) in the
directory of your choice.

### Download the pretrained models

Download the [few-shot](https://drive.google.com/file/d/1rTG7AjGmasfOYFm-ZzSbVQH9daYgOoIS/view?usp=sharing) and/or [zero-shot](https://drive.google.com/file/d/11-gkybBmBhQF2KZyo-c2-4IGUmor_JMu/view?usp=sharing) pretrained models and place them in a directory of your choice.

### Generate density maps (optional)

If you wish to train LOCA on the FSC147 dataset, you need to run the density map generation script as follows:

    python utils/data.py --data_path <path_to_your_data_directory> --image_size 512 
    
where `<path_to_your_data_directory>` is the path to the dataset created in the previous step.


## Training

The training code is located in the `train.py` script. It is adapted for distributed training on multiple GPUs. An example of how to run training on a SLURM-based system is shown in `train_few_shot.sh` and `train_zero_shot.sh`. Make sure to modify the paths in `--data_path` and `--model_path` to point to your data and model directories. To run locally on multiple GPUs, use `torchrun` (e.g., `torchrun main.py --nproc_per_node=2 train.py <LOCA args (see SLURM example)>`).

## Evaluation

To evaluate LOCA, use the `evaluate.py` script (used the same way as `train.py`). You might want to run the evaluation on a single GPU. You can do that for the provided pretrained models as follows:

#### few-shot:

```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 evaluate.py --model_name loca_few_shot --backbone resnet50 --swav_backbone --reduction 8 --image_size 512 --num_enc_layers 3 --num_ope_iterative_steps 3 --emb_dim 256 --num_heads 8 --kernel_dim 3 --num_objects 3 --pre_norm
```

#### zero-shot:

```
CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 evaluate.py --model_name loca_zero_shot --backbone resnet50 --swav_backbone --reduction 8 --image_size 512 --num_enc_layers 3 --num_ope_iterative_steps 3 --emb_dim 256 --num_heads 8 --kernel_dim 3 --num_objects 3 --pre_norm --zero_shot
```