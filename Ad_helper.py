import torch
import numpy 

def check_detect(detect_adv_idx, gt_adv_idx):
    intersection = [idx for idx in gt_adv_idx if idx in detect_adv_idx]
    if len(intersection) > 0:
        return True
    else:
        return False
    
# feature_matrix:
# each row is flatten dWs from a client
def generate_feature_matrix(dW_dicts):
    rows = []
    
    for dW_dict in dW_dicts.items():
        row = torch.empty(0)
        for key, value in dW_dict:
            row = torch.cat((row, value.flatten()), 0)
        rows.append(row)
        
    matrix = torch.stack(rows, 0)
    return matrix.numpy()