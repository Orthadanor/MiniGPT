"""
Training file for the models we implemented 
"""

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.utils
from torch.utils.data import DataLoader
from einops import rearrange
import wandb

from model import BigramLanguageModel, MiniGPT
from dataset import TinyStoriesDataset
from config import BigramConfig, MiniGPTConfig

def solver(model_name):
    # Initialize the model
    if model_name == "bigram":
        config = BigramConfig(log_interval=1000, batch_size=128, save_iterations=50000)
        model = BigramLanguageModel(config)
    elif model_name == "minigpt":
        config = MiniGPTConfig
        model = MiniGPT(config)
    else:
        raise ValueError("Invalid model name")
    
    # Load the dataset
    train_dataset = TinyStoriesDataset(
        config.path_to_data,
        mode="train",
        context_length=config.context_length,
    )
    eval_dataset = TinyStoriesDataset(
        config.path_to_data, mode="test", context_length=config.context_length
    )

    # Create the dataloaders
    train_dataloader = DataLoader( # len(train_dataloader) =  118398
        train_dataset, batch_size=config.batch_size, pin_memory=True, num_workers=4
    ) 
    eval_dataloader = DataLoader( # len(eval_dataloader) =  29600
        eval_dataset, batch_size=config.batch_size, pin_memory=True, num_workers=4
    ) 
    
    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Print number of parameters in the model
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of trainable parameters: %.2fM" % (count_parameters(model) / 1e6,))

    # Initialize wandb if you want to use it
    if config.to_log:
        wandb.init(project="dl2_proj3")

    # Create the save path if it does not exist
    if not Path.exists(config.save_path):
        Path.mkdir(config.save_path, parents=True, exist_ok=True)

    ### ==================== START OF YOUR CODE ==================== ###
    """
    You are required to implement the training loop for the model.

    The code below is a skeleton for the training loop, for your reference. 
    You can fill in the missing parts or completely set it up from scratch.

    Please keep the following in mind:
    - You will need to define an appropriate loss function for the model.
    - You will need to define an optimizer for the model.
    - You are required to log the loss (either on wandb or any other logger you prefer) every `config.log_interval` iterations.
    - It is recommended that you save the model weights every `config.save_iterations` iterations. You can also just save the model with the best training loss.

    NOTE : 
    - Please check the config file to see the different configurations you can set for the model.
    - The MiniGPT config has params that you do not need to use, these were added to scale the model but are 
    not a required part of the assignment. 
    - Feel free to experiment with the parameters and I would be happy to talk to you about them if interested.
    """

    ### ========= TODO : START ========= ###
    # Define the loss function
    loss_fn = nn.CrossEntropyLoss()

    # Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr = config.learning_rate)
    ### ======== TODO : END ========= ###

    if config.scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=2000, T_mult=2
        )

    model.train()
    model.to(device)
    
    best_eval_loss = 1e10 # Best eval loss to be updated
    print("len(train_dataloader) = ", len(train_dataloader))
    
    for i, (context, target) in enumerate(train_dataloader):

        train_loss = None # You can use this variable to store the training loss for the current iteration
        ### ======== TODO : START ========= ###
        # Do the forward pass, compute the loss, do the backward pass, and update the weights with the optimizer.
        
        context, target = context.to(device), target.to(device) # (batch_size, context_length=1)
        logits = model(context) # (batch_size, vocab_size)

        train_loss = loss_fn(logits, target.squeeze(1))
        
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        ### ======== TODO : END ========= ###

        if config.scheduler:
            scheduler.step()

        del context, target # Clear memory

        if i % config.log_interval == 0:

            model.eval()
            eval_loss = 0.0 # You can use this variable to store the evaluation loss for the current iteration
            ### ======== TODO : START ========= ###
            # Compute the evaluation loss on the eval dataset.
            
            print("len(eval_dataloader) = ", len(eval_dataloader))
            with torch.no_grad():
                for j, (context, target) in enumerate(eval_dataloader):
                    
                    if j > len(eval_dataloader):
                        break
                    
                    context, target = context.to(device), target.to(device) # (batch_size, context_length=1)
                    logits = model(context)
                    eval_loss = eval_loss + loss_fn(logits, target.squeeze(1)).item()
            eval_loss /= len(eval_dataloader)
            
            ### ======== TODO : END ========= ###
            
            print(
                f"Iteration {i}, Train Loss: {train_loss.item():.4f}",
                f"Eval Loss: {eval_loss:.4f}",
            )
            # Log the loss using wandb every `config.log_interval` iterations.
            if config.to_log:
                wandb.log(
                    {
                        "Train Loss": train_loss,
                        "Eval Loss": eval_loss,
                        "Learning Rate": scheduler.get_last_lr()[0] if config.scheduler else config.learning_rate
                    }
                )

            model.train()

            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": train_loss,
                        "eval_loss": eval_loss,
                        "iteration": i,
                    },
                    config.save_path / "best_model.pt",
                )
        
        # Save the model every config.save_iterations
        if i % config.save_iterations == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "eval_loss": eval_loss,
                    "iteration": i,
                },
                config.save_path / f"mini_model_checkpoint_{i}.pt",
            )

        if i > config.max_iter:
            break