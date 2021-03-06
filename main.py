from torchvision import transforms
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.multiprocessing import freeze_support
import time

from preprocessing import *
from make_dataset import *
from model import *
from save_load import *
from postprocessing import *
from func import sample_condition_generate
### hyperparameter
############################################################################################################################# 
DEVICE = torch.device("cuda:0")
#DEVICE = torch.device("cpu")


PROCESS_DATA = False
DATA_PATH = r'C:/USING_DATA/DATASET/pix2pix-dataset/edges2shoes/edges2shoes/train'
PROCESSED_DATA_ROOT = r'C:/USING_DATA/PROCESSED_DATA/pix2pix/train'
FILE_PATH_LIST_PATH = r'C:/USING_DATA/PROCESSED_DATA/pix2pix/variables/file_path_list.file'
FIXED_PATH = r'C:/USING_DATA/DATASET/pix2pix-dataset/edges2shoes/edges2shoes/val'

SAVE_MODEL = True
LOAD_MODEL = True
G_PATH = r'C:/USING_DATA/MODEL/pix2pix/generator.model'
D_PATH = r'C:/USING_DATA/MODEL/pix2pix/discriminator.model'

TEST_MODE = False
if TEST_MODE:
    LOAD_MODEL = True

BATCH_SIZE = 4
SHUFFLE = True
NUM_WORKERS = 0 # multithreading

EPOCH = 200
NUM_LEARN_D = 1
NUM_LEARN_G = 1
G_LEARNING_RATE = 0.0002
D_LEARNING_RATE = 0.0002

"""
G_WIDTH = None
G_LENGTH = None
D_WIDTH = None
D_LENGTH = None
"""
#############################################################################################################################


### preprocessing, make or load and save dataset
############################################################################################################################# 
filter = transform_func(real_mode='jpg', condition_mode='jpg')
real_process = [filter.first_processing_real, filter.to_FloatTensor, filter.Scaling, filter.final_processing]
condition_process = [filter.first_processing_condition, filter.to_FloatTensor, filter.Scaling, filter.final_processing]

if PROCESS_DATA:

    ### preprocessing
    data_path = DATA_PATH
    sample_list = PreProcessing(data_path) 

    ### make dataset
    ##############################################################################
    transform = my_transform(real_process=real_process, condition_process=condition_process)
    file_path_list = Transform_Processing(sample_list, transform, PROCESSED_DATA_ROOT)

    with open(FILE_PATH_LIST_PATH, 'wb') as f:
        pickle.dump(file_path_list, f)
    ###############################################################################

else:

    with open(FILE_PATH_LIST_PATH, 'rb') as f:
        file_path_list = pickle.load(f)

dataset = PathDataset(file_path_list)

### make dataloader
dataloader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle=SHUFFLE, num_workers=NUM_WORKERS)
#######################################################################################################################


### model information
###########################################
"""
print(model)
params = list(model.parameters())
print(len(params))
print(params[0].size())
"""
###########################################


### model definition
#################################
if LOAD_MODEL:
    G = load_model_state(Generator, G_PATH).to(DEVICE)
    D = load_model_state(Discriminator, D_PATH).to(DEVICE)
else:
    G = Generator().to(DEVICE)
    D = Discriminator().to(DEVICE)

G_OPTIMIZER = Adam(G.parameters(), lr=G_LEARNING_RATE)
D_OPTIMIZER = Adam(D.parameters(), lr=D_LEARNING_RATE)
#################################


# fixed vector for visualization
##################################
# fixed latent vector
"""
fixed_z = torch.randn((100,100)).to(DEVICE)
"""
# fixed condition
"""
idx = filter.to_LongTensor([[element] for element in range(10)])
fixed_condition = [idx for _ in range(10)]
fixed_condition = torch.cat(fixed_condition) # shape:(100,1)
"""
fixed_condition = sample_condition_generate(data_path=FIXED_PATH, num=25, condition_process=condition_process, device=DEVICE)
##################################


### training, evaluate, log and model save
##########################################################################################
if __name__ == '__main__': #for multiprocessing

    epoch = range(EPOCH)
    for times in epoch:
        start_time = time.time()

        D_losses = []
        G_losses = []
        batch_len = len(dataloader)

        for data in dataloader:

            data['real'] = data['real'].to(DEVICE)
            data['condition'] = data['condition'].to(DEVICE) # some trick

            ### fitting batch size
            batch_size = data['condition'].size(0)
            REAL_ANSWER = torch.ones([batch_size, 1, 16, 16]).to(DEVICE)
            FAKE_ANSWER = torch.zeros([batch_size, 1, 16, 16]).to(DEVICE)

            G.train()
            D.train()
            ### train discriminator
            for i in range(NUM_LEARN_D):
                D_input_real = D_input_processing(D, DEVICE, data['real'], data['condition'])
                D_result_real = D(D_input_real)

                G_input = G_input_processing(G, DEVICE, data['condition'])
                fake_data = G(G_input)

                #visualization(fake_data,'./files/fake_data.jpg',mode='RGB') just for test

                D_input_fake = D_input_processing(D, DEVICE, fake_data, data['condition'])
                D_result_fake = D(D_input_fake)

                D_loss_args = {'D_result_real': D_result_real, 'D_result_fake': D_result_fake, 'real_answer': REAL_ANSWER, 'fake_answer': FAKE_ANSWER}

                D_loss, D_loss_real, D_loss_fake = D_CRITERION(**D_loss_args)
                D_losses.append(D_loss.data)

                D.zero_grad()
                D_loss.backward()
                D_OPTIMIZER.step()
            
            ### train generator
            for i in range(NUM_LEARN_G):
                G_input = G_input_processing(G, DEVICE, data['condition'])
                fake_data = G(G_input)

                D_input_fake = D_input_processing(D, DEVICE, fake_data, data['condition'])
                D_result_fake = D(D_input_fake)

                G_loss_args = {'D_result_fake': D_result_fake, 'real_answer': REAL_ANSWER, 'real_data': data['real'], 'fake_data': fake_data, 'lambda': 100}

                G_loss = G_CRITERION(**G_loss_args)
                G_losses.append(G_loss.data)

                G.zero_grad()
                G_loss.backward()
                G_OPTIMIZER.step()

        ### log
        #####################################
        end_time = time.time()
        running_time = int(end_time - start_time)
        hour = int(running_time / 3600)
        minute = int((running_time - hour*3600) / 60)
        second = running_time - hour*3600 - minute*60

        print("Epoch: " + str(times))
        print("Time: " + str(hour) + 'h' + '   ' + str(minute) + 'm' + '   '+ str(second) + 's')
        print("Average G Loss:", float(sum(G_losses)/len(G_losses)), "     Average D Loss", float(sum(D_losses)/len(D_losses)), "\n")
        #####################################

        ### visualization 
        #################################
        #G.eval()
        fixed_G_input = G_input_processing(G, DEVICE, fixed_condition, mode='val')
        generated_sample = G(fixed_G_input) # shape: (Batch_size, 3, 256, 256)

        ## concat condition and output image
        fixed_condition__ = torch.cat([fixed_condition,fixed_condition,fixed_condition], dim=1)
        generated_sample = torch.cat([fixed_condition__, generated_sample], dim=3)

        path = './result/epoch ' + str(times) + '.jpg'
        visualization(generated_sample,path,mode='RGB')
        ##################################

        ### model save
        if SAVE_MODEL:
            save_model_state(G.state_dict(), G_PATH)
            save_model_state(D.state_dict(), D_PATH)
############################################################################################

### 테스트용 코드를 아예 따로 짜자...