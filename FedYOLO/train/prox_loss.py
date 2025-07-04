import torch
from ultralytics.utils.loss import v8DetectionLoss

class ProximalDetectionLoss(v8DetectionLoss):
    """Criterion class for computing training losses with proximal regularization for federated learning."""

    def __init__(self, model, global_params, proximal_mu):
        
        """
        Initialize the ProximalDetectionLoss with model and proximal term parameters.
        
        Args:
            model: YOLO model
            global_params: Dictionary containing the global model parameters
            proximal_mu: Weight of the proximal term (default: 0.1)
            tal_topk: Top-k for task-aligned assigner
        """
        
        super().__init__(model)
        self.global_params = global_params 
        self.proximal_mu = proximal_mu
        self.model = model
        print(f"[LOSS] ProximalDetectionLoss initialized with proximal_mu: {self.proximal_mu}")


    def __call__(self, preds, batch):
        
        """Calculate the sum of the task loss and proximal term."""

        # Get the original task loss
        loss, loss_items = super().__call__(preds, batch)
        
        # Add proximal term if global parameters are available
        proximal_loss = torch.tensor(0.0, device=self.device)

        if self.global_params:
            # current_params = self.model.state_dict()
            
            # Calculate the proximal term
            for name, param in self.model.named_parameters():
                if name in self.global_params:
                    proximal_loss += torch.sum(torch.square(param - self.global_params[name].to(param.device)))
            
            # Scale by proximal_mu/2
            proximal_loss = 0.5 * self.proximal_mu * proximal_loss 
            
            # Add to task loss
            total_loss = loss + proximal_loss

            loss_items = torch.cat([loss_items, proximal_loss.unsqueeze(0)])

        print(f"[LOSS] Task loss: {task_loss.item()}, Proximal loss: {proximal_loss.item()} ")
        return total_loss, loss_items
    
    def update_global_params(self, global_params):
        
        """Update the global parameters used for the proximal term."""

        self.global_params = global_params
