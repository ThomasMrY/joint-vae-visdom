import torch
from jointvae.models import VAE
from jointvae.training import Trainer
from utils.dataloaders import get_dsprites_dataloader,get_mnist_dataloaders
from utils.load_model import load_param
from torch import optim

dataset = "mnist"
load_data = False
viz_on = False
num = 3

path = './trained_models/'+dataset+'/'
model_path = './trained_models/'+dataset+'/model'+str(num)+'.pt'
spec,img_size = load_param(path)
print(spec)
print("Training Start!!! :{}".format(num))

batch_size = spec['batch_size']
lr = spec['lr'][0]
epochs = spec['epochs'][0]

# Check for cuda
use_cuda = torch.cuda.is_available()

# Load data
data_loader,_= get_mnist_dataloaders(batch_size=batch_size)

# Define latent spec and model
latent_spec = spec['latent_spec']
model = VAE(img_size=img_size, latent_spec=latent_spec,
            use_cuda=use_cuda)
if use_cuda:
    model.cuda()

# Define optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Define trainer
trainer = Trainer(model, optimizer,
                  cont_capacity=spec['cont_capacity'],
                  disc_capacity=spec['disc_capacity'],
                  spec=spec,
                  viz_on = viz_on,
                  use_cuda=use_cuda)

# Train model for 100 epochs
trainer.train(data_loader, epochs)

# Save trained model
torch.save(trainer.model.state_dict(), model_path)
print("Training finished!!! :{}".format(num))