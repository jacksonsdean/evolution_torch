"""Autoencoder for novelty measurement."""

from collections import OrderedDict
from typing import List
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
# from image_utils import *
import gc 
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping
from torch import nn
import logging

"""
    We explore two features which define the dimensions of the map.
    The first, complexity, is simply the number of connections in the
    network. For the second, we introduce a generic notion of novelty.
    Periodically all of the images in the map are compiled and used
    to train an autoencoder [3]; when a new image is produced we
    compress and reconstruct the image with the autoencoder, examining the pixel-by-pixel error of the reconstruction. Images similar
    to those which were used to train the autoencoder will be reconstructed well, those different poorly. The reconstruction error is
    used as generic approximation of “novelty” or “interestingness”.
    
    - Adam Gaier, Alexander Asteroth, and Jean-Baptiste Mouret. 2019
    
    [3] G Hinton and R Salakhutdinov. 2006. Reducing the dimensionality of data with
        neural networks. Science (2006).
"""
class AutoEncoder(LightningModule):
    instance = None
    def __init__(self) -> None:
        super().__init__()
        logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)
        
        self.BATCH_SIZE = 32
        self.EPOCHS = 1
        self.AVAIL_GPUS = min(1, torch.cuda.device_count())
        
    def setup_novelty(self, input_shape, original_dims):
        
        self.encoder_hidden_layer = nn.Linear(
            in_features=input_shape, out_features=256
        )
        self.encoder_output_layer = nn.Linear(
            in_features=256, out_features=128
        )
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=256
        )
        self.decoder_output_layer = nn.Linear(
            in_features=256, out_features=input_shape
        )
        self.configure_optimizers()
        self.criterion = nn.MSELoss()
        self.ready = False
        self.original_dims = original_dims
        self.input_shape = input_shape
        AutoEncoder.instance = self
        return self
        
    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        return reconstructed
    
    def training_step(self, batch, batch_idx):
        batch = batch.flatten(start_dim=1).float()
        x = batch
        device = self.get_device(x)
        outputs = self(x) # compute reconstructions
        loss = self.criterion(outputs, x) # compute training reconstruction loss
      
        # compute the epoch training loss
        loss = loss / self.BATCH_SIZE
        log = {
            "train_loss": loss,
        }
        status = {
            "steps": torch.tensor(self.global_step).to(device),
        }
        return OrderedDict({"loss": loss, "log": log, "progress_bar": status})
    
    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"
    
    def eval_image(self, image):
        self.eval()
        input_shape=image.numel()
        images_tensor = image
        images_tensor = images_tensor.unsqueeze(0)
        images_tensor = images_tensor.view(-1, input_shape) # flatten
        if images_tensor.max() > 1:
            images_tensor = images_tensor / 255.0
        images_tensor = self(images_tensor)
        return images_tensor
    
    def eval_images(self, images):
        self.eval()
        input_shape=len(images[0].flat)
        images_tensor = tensor(images).float()#.to(device)
        images_tensor = images_tensor.view(-1, input_shape)
        if images_tensor.max() > 1:
            images_tensor = images_tensor / 255.0
        output = self(images_tensor)
        return output
    
    def tensor_to_np(self, ten):
        image = torch.reshape(ten, self.original_dims)
        image = image.detach().cpu().numpy()
        return image
    
    def np_to_tensor(self, numpy_array):
        if isinstance(numpy_array, torch.Tensor):
           return numpy_array
        return torch.tensor(numpy_array.astype(np.float32))#.to(device)
    
    def eval_image_novelty(self, image):
        input_shape=len(image.flat)
        encoded = self.eval_image(image) # send through auto-encoder
        image = image.unsqueeze(0)
        images_tensor = images_tensor.view(-1, input_shape)
        reconstruction_error = self.criterion(encoded, images_tensor).detach() # compute reconstruction loss
        return reconstruction_error
    
    def eval_images_novelty(self, images):
        input_shape=images[0].numel()
        images_tensor = torch.stack(images).to(self.device)
        images_tensor = images_tensor.view(-1, input_shape).float()
        if images_tensor.max() > 1:
            images_tensor = images_tensor / 255.0
        encoded = self(images_tensor) # send through auto-encoder
        error_criterion = nn.MSELoss(reduction='none')
        reconstruction_errors = torch.mean(error_criterion(encoded, images_tensor), dim=1).detach() # compute reconstruction loss
        return reconstruction_errors
    
    def eval_image_fitness(self, image):
        input_shape=len(image.flat)
        encoded = self.eval_image(image) # send through auto-encoder
        image = np.expand_dims(image, axis=0) # convert to batch
        image = torch.tensor(image)#.to(device)
        image = image.view(-1, input_shape)
        reconstruction_error = self.criterion(encoded, image).detach().cpu().numpy() # compute training reconstruction loss
        return 1.0-reconstruction_error
    
    def update_novelty_network(self, population):
        self.train()
        
        self.train_loader = torch.utils.data.DataLoader(
            [self.np_to_tensor(g.image) for g in population], batch_size=self.BATCH_SIZE, shuffle=False, num_workers=0
        )
        early_stop_callback = EarlyStopping(monitor="train_accuracy", min_delta=0.00, patience=3, verbose=False, mode="max")
        
        trainer = Trainer(
            gpus=self.AVAIL_GPUS,
            max_epochs=self.EPOCHS,
            val_check_interval=0.0,
            callbacks=[early_stop_callback],
        )
        trainer.fit(self,  self.train_loader)
        
        self.ready = True
    
    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return [ self.optimizer ]

    def __dataloader(self) -> DataLoader:
        return  self.train_loader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()
    
    def get_ae_novelties(self, population):
        return self.eval_images_novelty([i.image for i in population])
        

def initialize_encoders(config, target):
    novelty_ae = AutoEncoder()
    device = config.device
    if(isinstance(target, str)): # for classification
        input_shape = config.classification_image_size[0] * config.classification_image_size[1] * len(config.color_mode)
        dims = (config.classification_image_size[0], config.classification_image_size[1], len(config.color_mode))
        AutoEncoder.instance = novelty_ae.setup_novelty(input_shape, dims).to(device)
    else:
        AutoEncoder.instance = novelty_ae.setup_novelty(target.numel(), original_dims=target.shape).to(device)
        