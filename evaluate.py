from models.loca import build_model
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser

import argparse
import os

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist


@torch.no_grad()
def evaluate(args):

    if 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))['model']
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    for split in ['val', 'test']:
        test = FSC147Dataset(
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
        )
        test_loader = DataLoader(
            test,
            sampler=DistributedSampler(test),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()
        for img, bboxes, density_map in test_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            density_map = density_map.to(device)

            out, _ = model(img, bboxes)
            ae += torch.abs(
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ).sum()
            se += ((
                density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            ) ** 2).sum()

        dist.all_reduce_multigpu([ae])
        dist.all_reduce_multigpu([se])

        if rank == 0:
            print(
                f"{split.capitalize()} set",
                f"MAE: {ae.item() / len(test):.2f}",
                f"RMSE: {torch.sqrt(se / len(test)).item():.2f}",
            )

    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
    evaluate(args)
