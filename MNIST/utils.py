import torch

def convert_targets(targets, task_obj):
    return torch.tensor([target.item() in task_obj['Task'] for target in targets], \
                            dtype=torch.float32)
