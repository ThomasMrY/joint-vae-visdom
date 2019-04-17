import torch
import multiprocessing
from jointvae.models import VAE
from jointvae.training import Trainer
from utils.dataloaders import get_dsprites_dataloader,get_mnist_dataloaders
from utils.load_model import load_param
from torch import optim
data_loader,_= get_mnist_dataloaders(batch_size=256)
Acc_list = []
def training_process(num):
    dataset = "mnist"
    load_data = False
    viz_on = False

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
    print("Use_cuda:%s"%use_cuda)
    # Load data
    data_loader,_= get_mnist_dataloaders(batch_size=batch_size)

    # Define latent spec and model
    latent_spec = spec['latent_spec']
    locals()['model_'+str(i)] = VAE(img_size=img_size, latent_spec=latent_spec,
                use_cuda=use_cuda)
    if use_cuda:
        locals()['model_'+str(i)].cuda()

    # Define optimizer
    optimizer = optim.Adam(locals()['model_'+str(i)].parameters(), lr=lr)

    # Define trainer
    trainer = Trainer(locals()['model_'+str(i)], optimizer,
                      cont_capacity=spec['cont_capacity'],
                      disc_capacity=spec['disc_capacity'],
                      spec=spec,
                      viz_on = viz_on,
                      use_cuda=use_cuda,
                      num = num)

    # Train model for 100 epochs
    acc = trainer.train(data_loader, epochs)
    Acc_list.append(acc)
    # Save trained model
    torch.save(trainer.model.state_dict(), model_path)
    print("Training finished!!! :{}".format(num))

processes = []

for i in range(9):
    p = multiprocessing.Process(target=training_process , args=(i, ))
    processes.append(p)
    p.start()

for p in processes:
    p.join()
print("The final acc:\n")
print(Acc_list)
print("The highest acc ex:\n")
print(Acc_list.index(max(Acc_list)))