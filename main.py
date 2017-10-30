import tensorflow as tf
import numpy as np

from model.ImageCaptioning import ImageCaptioning
from operations.train import TrainCaptionModel
from configuration.config import ModelConfig, TrainingConfig

## Load Configuration
print("Loading Model Configuration")
model_config = ModelConfig()
train_config = TrainingConfig()

## Create Model
print("Setting up model")
model = ImageCaptioning(model_config,"train")
model.build()
print("Model Loaded")

print("Traing Model")
train = TrainCaptionModel(model,train_config)
train.train_model(restore_model=True)
