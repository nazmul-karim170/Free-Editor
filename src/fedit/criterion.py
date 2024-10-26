import torch
import torch.nn as nn
from utils import img2mse
import torch.nn.functional as F

class Criterion(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, outputs, ray_batch, scalars_to_log):
        """
        training criterion
        """
        pred_rgb = outputs["rgb"]
        if "mask" in outputs:
            pred_mask = outputs["mask"].float()
        else:
            pred_mask = None
        gt_rgb = ray_batch["rgb"]

        ## MSE Loss 
        loss = img2mse(pred_rgb, gt_rgb, pred_mask)

        ## Self-view robust loss


        ## Multi-view consistency loss 


        return loss, scalars_to_log




    def jensen_shannon_divergence(self, p, q):
        """
        Compute the Jensen-Shannon Divergence between two distributions using PyTorch.
        
        Args:
        - p: Torch tensor of shape (batch_size, feature_dim).
        - q: Torch tensor of shape (batch_size, feature_dim).
        
        Returns:
        - jsd: Torch tensor, the Jensen-Shannon divergence.
        """
        # Normalize the distributions
        p = p / torch.sum(p, dim=-1, keepdim=True)
        q = q / torch.sum(q, dim=-1, keepdim=True)
        
        # Compute the average distribution
        m = 0.5 * (p + q)
        
        # Compute KL divergence and then JSD
        kl_pm = F.kl_div(m.log(), p, reduction='batchmean')
        kl_qm = F.kl_div(m.log(), q, reduction='batchmean')
        jsd = 0.5 * (kl_pm + kl_qm)
        
        return jsd

    def multi_view_consistency_loss(self, points_j, points_j_prime, features_j, features_j_prime):
        """
        Compute the multi-view consistency loss using PyTorch.
        
        Args:
        - points_j: Tensor of shape (P, 3) representing 3D points along ray r_j.
        - points_j_prime: Tensor of shape (P, 3) representing 3D points along ray r_j'.
        - features_j: Tensor of shape (P, feature_dim) representing features along r_j.
        - features_j_prime: Tensor of shape (P, feature_dim) representing features along r_j'.
        
        Returns:
        - L_con: The multi-view consistency loss (scalar).
        """
        # Compute the Euclidean distance between corresponding points along the rays
        d_p_j_j_prime = torch.norm(points_j - points_j_prime, dim=-1)  # (P,)
        
        # Compute the Jensen-Shannon Divergence between the feature distributions
        L_J = self.jensen_shannon_divergence(features_j, features_j_prime)  # Scalar
        
        # Compute weights based on Euclidean distances
        weights = torch.exp(-d_p_j_j_prime)
        
        # Normalize the weights
        weights = weights / torch.sum(weights)
        
        # Compute the final weighted loss
        L_con = torch.sum(weights * L_J)
        
        return L_con







# # Example usage (on GPU):
# # Assuming you have your data as torch tensors on the GPU
# device = torch.device('cuda')

# # Define some sample points along ray r_j and r_j' and their corresponding features
# points_j = torch.rand(10, 3, device=device)  # 10 points in 3D space on GPU
# points_j_prime = torch.rand(10, 3, device=device)  # 10 nearby points in 3D space on GPU

# # Define random feature distributions for these points (assuming feature dimension of 5)
# features_j = torch.rand(10, 5, device=device)  # Features on GPU
# features_j_prime = torch.rand(10, 5, device=device)  # Features on GPU

# # Compute the multi-view consistency loss
# L_con = multi_view_consistency_loss(points_j, points_j_prime, features_j, features_j_prime)
# print("Multi-view consistency loss:", L_con.item())
