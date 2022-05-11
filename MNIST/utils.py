import torch

def convert_targets(targets, task_obj):
    tmp = torch.tensor([task_obj['Task'].index(target.item()) if target.item() in task_obj['Task'] \
           else task_obj['Output_Dim']-1 for target in targets])
    return (torch.nn.functional.one_hot(tmp, num_classes=task_obj['Output_Dim'])).to(dtype=torch.float32)
    
# require targets to be already-converted one-hot encoding
def compute_accuracy(predictions, targets):
    assert targets.ndim == 2
    assert predictions.size() == targets.size()
    return (predictions.argmax(dim=1) == targets.argmax(dim=1)).to(dtype=torch.float32).mean()
