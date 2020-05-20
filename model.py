import torch
import torch.nn as nn
import torchvision as tv
# final input processing
########################################################################################
def G_input_processing(model, device, condition, latent=None, mode='train'):
    G_input = condition
    return G_input


def D_input_processing(model, device, real, condition):
    if real.device.type == 'cpu':
        real = real.to(device)
    
    D_input = torch.cat([real, condition], 1) #
    
    return D_input
#########################################################################################


# submodel
##############################################################
class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x
##############################################################


# Generator and Discriminator
###############################################################
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        
        self.down1 = UNetDown(1, 64, normalize=False)
        self.down2 = UNetDown(64, 128)
        self.down3 = UNetDown(128, 256)
        self.down4 = UNetDown(256, 512, dropout=0.5)
        #self.down5 = UNetDown(512, 512, dropout=0.5)
        #self.down6 = UNetDown(512, 512, dropout=0.5)
        #self.down7 = UNetDown(512, 512, dropout=0.5)
        #self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        #self.up1 = UNetUp(512, 512, dropout=0.5)
        #self.up2 = UNetUp(1024, 512, dropout=0.5)
        #self.up3 = UNetUp(1024, 512, dropout=0.5)
        #self.up4 = UNetUp(512, 512, dropout=0.5)
        self.up5 = UNetUp(512, 256)
        self.up6 = UNetUp(512, 128)
        self.up7 = UNetUp(256, 64)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(128, 3, 4, padding=1),
            nn.Tanh(),
        )
            # output shape: (B,3,256,256)
            # downsampling에서 줄여나가고 upsampling은 transposeconv를 쓰는게 아니라 upsampling후 conv를 사용
    
    def forward(self, G_input):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(G_input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        #d5 = self.down5(d4)
        #d6 = self.down6(d5)
        #d7 = self.down7(d6)
        #d8 = self.down8(d7)
        #u1 = self.up1(d8, d7)
        #u2 = self.up2(u1, d6)
        #u3 = self.up3(u2, d5)
        #u4 = self.up4(d5, d4)
        u5 = self.up5(d4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        G_out =  self.final(u7)        
        return G_out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def discriminator_block(in_filters, out_filters, normalization=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalization:
                layers.append(nn.InstanceNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(4, 64, normalization=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(512, 1, 4, padding=1, bias=False)
        )
        
    def forward(self, D_input):
        D_out = self.model(D_input)
        return D_out
#############################################################


# calculate loss
#############################################################
def G_CRITERION(**kwargs):
    loss_function_1 = nn.MSELoss()
    loss_function_2 = nn.L1Loss()

    G_conditional_loss = loss_function_1(kwargs['D_result_fake'], kwargs['real_answer'])
    G_L1_loss = loss_function_2(kwargs['fake_data'], kwargs['real_data'])

    G_loss = G_conditional_loss + kwargs['lambda']*G_L1_loss
    return G_loss


def D_CRITERION(**kwargs):
    loss_function = nn.MSELoss()
    D_loss_real = loss_function(kwargs['D_result_real'], kwargs['real_answer'])
    D_loss_fake = loss_function(kwargs['D_result_fake'], kwargs['fake_answer'])
    D_loss = 0.5 * (D_loss_real + D_loss_fake)

    return D_loss, D_loss_real, D_loss_fake
##########################################################





    
        
