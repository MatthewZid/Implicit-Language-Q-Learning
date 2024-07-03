from torch.utils.data.dataset import IterableDataset
from load_objects import load_item
from accelerate import Accelerator
from utils.misc import add_system_configs
from data.torch_datasets import GeneralDataset, GeneralIterDataset
from data.rl_data import Iterable_RL_Dataset
from torch.utils.data import DataLoader
import torch
from utils.torch_utils import to
from tqdm.auto import tqdm
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

def inference(cfg):
    print('using config:', cfg)
    eval_cfg = cfg['eval']
    accelerator = Accelerator()
    system_cfg = add_system_configs(cfg, accelerator)
    print('using device:', system_cfg['device'])
    print('num processes:', system_cfg['num_processes'])
    print('using fp16:', system_cfg['use_fp16'])
    
    raw_dataset = load_item(cfg['eval_dataset'], system_cfg['device'])

    if isinstance(raw_dataset, Iterable_RL_Dataset):
        dataset = GeneralIterDataset(raw_dataset, 'cpu')

    else:
        # raw_dataset_train -> General Dataset
        dataset = GeneralDataset(raw_dataset, 'cpu')
    
    data_loader_kwargs = {'num_workers': eval_cfg['dataloader_workers'], 
                                'batch_size': eval_cfg['bsize'], 
                                'collate_fn': dataset.collate}
    
    if not isinstance(dataset, IterableDataset):
        data_loader_kwargs['shuffle'] = True
    
    data_loader = DataLoader(dataset, **data_loader_kwargs)
    model = load_item(cfg['model'], system_cfg['device'])

    if isinstance(dataset, IterableDataset):
        model = accelerator.prepare(model)
    else:
        model, data_loader = accelerator.prepare(model, data_loader)
    model.eval()

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(f'/home/vasters/titan-rl/Implicit-Language-Q-Learning/scripts/eval/runs/eval_bc_{timestamp}')

    with torch.no_grad():
        for i, eval_items in tqdm(enumerate(data_loader)):
            eval_items = to(eval_items, system_cfg['device'])
            if eval_items['tokens'].size(1) >= 1024: continue

            if i >= eval_cfg['batches']:
                break

            loss, _, _ = accelerator.unwrap_model(model).get_loss(eval_items, **eval_cfg['loss'])
            writer.add_scalar("Loss/inference", loss, i+1)
            writer.flush()

        writer.close()