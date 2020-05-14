import torch
import torch.nn as nn
import torchvision as tv

def G_input_processing(model, device, condition, latent=None, mode='train'):
    condition_vector = torch.zeros((condition.size(0),10))
    for i in range(condition.size(0)):
        condition_vector[i][condition[i].data] = 1  
    condition_vector = condition_vector.to(device)

    if condition is not None:
        batch_size = condition.size(0)
        z = torch.randn((batch_size,100)).to(device)

        if mode=='val':
            z = latent

        G_input = torch.cat((condition_vector.view(batch_size,-1), z), -1)

    return G_input


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(110,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512,1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024,784),
            nn.Tanh()
        )
    
    def forward(self, G_input):
        G_out = self.model(G_input)
        return G_out


def D_input_processing(model, device, data, condition):
    #D_input = torch.cat((real_data, condition), 1)
    data = data.to(device)
    
    condition_vector = torch.zeros((condition.size(0),10))
    for i in range(condition.size(0)):
        condition_vector[i][condition[i].data] = 1    
    condition_vector = condition_vector.to(device)

    if condition is not None:
        batch_size = condition.size(0)
        D_input = torch.cat((condition_vector.view(batch_size,-1), data), -1)

    else:
        None
    
    return D_input


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(794,1024),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.5),

            nn.Linear(1024,512),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.5),

            nn.Linear(512,256),
            nn.LeakyReLU(0.2,inplace=True),
            nn.Dropout(0.5),

            nn.Linear(256,1),
            nn.Sigmoid()
        )
    
    def forward(self, D_input):
        D_out = self.model(D_input)
        return D_out


def G_CRITERION(D_result_fake, real_answer=None):
    loss_function = nn.BCELoss()
    G_loss = loss_function(D_result_fake, real_answer)
    return G_loss


def D_CRITERION(D_result_real, D_result_fake, real_answer=None, fake_answer=None):
    loss_function = nn.BCELoss()
    D_loss_real = loss_function(D_result_real, real_answer)
    D_loss_fake = loss_function(D_result_fake, fake_answer)
    D_loss = D_loss_real + D_loss_fake

    return D_loss, D_loss_real, D_loss_fake





    
        
