import torch
from torch import nn
import pytorch_lightning as pl
from models.pixelsnail import PixelSNAIL

class BottomSNAIL(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.model = PixelSNAIL(
            attention=False,
            input_channels=params.input_channels,
            n_codes=params.n_codes,
            n_snail_blocks=params.n_snail_blocks,
            n_res_blocks=params.n_res_blocks,
            n_filters=params.n_filters,
            condition=True,
            n_res_condition=params.n_res_condition
        )

        self.lr = params.lr
        self.criterion = nn.NLLLoss()
    
    def configure_optimizeres(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer
    
    def training_step(self, train_batch, batch_idx):
        loss = self.find_loss(train_batch, batch_idx)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        loss = self.find_loss(val_batch, batch_idx)
        self.log('val_loss', loss)
    
    def find_loss(self, batch, idx):
        top_code, bottom_code = batch
        target = torch.argmax(bottom_code, dim=1)
        pred = self.forward(bottom_code, cond=top_code)
        loss = self.criterion(pred, target)
        return loss
    
    def forward(self, bottom_code, cond=None):
        return self.model(bottom_code, cond=cond)