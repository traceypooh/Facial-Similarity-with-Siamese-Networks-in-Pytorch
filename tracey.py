import operator
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt  #tracey comment this and a few out if serverside only
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps

import torch.nn as nn
from torch import optim
import torch.nn.functional as F



class Config():
    training_dir = "./training/"
    testing_dir = "./testing/"
    train_batch_size = 64
    train_number_epochs = 200



def imshow(img, text):
    #return # tracey
    npimg = img.numpy()
    plt.axis("off")
    if text:
        plt.text(75, 8, text, style='italic',fontweight='bold',
            bbox={'facecolor':'white', 'alpha':0.8, 'pad':10})
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    print('  [paused on popup]')
    plt.show()

def show_plot(iteration,loss):
    #return # tracey
    plt.plot(iteration,loss)
    print('  [paused on popup]')
    plt.show()




class SiameseNetworkDataset(Dataset):

    def __init__(self, dir):
        self.imageFolderDataset = dset.ImageFolder(root = dir)
        self.transform = transforms.Compose([transforms.Scale((100,100)), transforms.ToTensor()])


    def __getitem__(self,index):
        img0_tuple = random.choice(self.imageFolderDataset.imgs)
        #we need to make sure approx 50% of images are in the same class
        should_get_same_class = random.randint(0,1)
        if should_get_same_class:
            while True:
                #keep looping till the same class image is found
                img1_tuple = random.choice(self.imageFolderDataset.imgs)
                if img0_tuple[1]==img1_tuple[1]:
                    break
        else:
            img1_tuple = random.choice(self.imageFolderDataset.imgs)

        # print(img0_tuple[0], " -VTRACEY- ", img1_tuple[0])

        img0 = Image.open(img0_tuple[0])
        img1 = Image.open(img1_tuple[0])
        img0 = img0.convert("L")
        img1 = img1.convert("L")

        img0 = self.transform(img0)
        img1 = self.transform(img1)

        return img0, img1, torch.from_numpy(np.array([int(img1_tuple[1]!=img0_tuple[1])],dtype=np.float32)), img0_tuple[0], img1_tuple[0]

    def __len__(self):
        return len(self.imageFolderDataset.imgs)






class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),

            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.Dropout2d(p=.2),
        )

        self.fc1 = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 500),
            nn.ReLU(inplace=True),

            nn.Linear(500, 5))

    def forward_once(self, x):
        output = self.cnn1(x)
        output = output.view(output.size()[0], -1)
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2




class ContrastiveLoss(torch.nn.Module):
    """
    Contrastive loss function.
    Based on: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    """

    def __init__(self, margin=2.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive




def training():
    train_dataloader = DataLoader(SiameseNetworkDataset(Config.training_dir),
                            shuffle=True,
                            num_workers=8,
                            batch_size=Config.train_batch_size)

    net = SiameseNetwork().cpu() #.cuda() #tracey
    criterion = ContrastiveLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )

    counter = []
    loss_history = []
    iteration_number = 0

    for epoch in range(0,Config.train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            img0, img1, label, file0, file1 = data
            # img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda() , Variable(label).cuda() #tracey
            img0, img1, label = Variable(img0).cpu(), Variable(img1).cpu(), Variable(label)
            output1,output2 = net(img0,img1)
            optimizer.zero_grad()
            loss_contrastive = criterion(output1, output2, label)
            loss_contrastive.backward()
            optimizer.step()
            if i %10 == 0 :
                print("Epoch {}.  Current loss: {}\n".format(epoch, loss_contrastive.data[0]))
                iteration_number +=10
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.data[0])
    show_plot(counter,loss_history)

    return net



def testing(net):
    test_dataloader  = DataLoader(SiameseNetworkDataset(Config.testing_dir),  shuffle=False, num_workers=6, batch_size=1)
    train_dataloader = DataLoader(SiameseNetworkDataset(Config.training_dir), shuffle=False, num_workers=6, batch_size=1)

    n = 0
    for j, data0 in enumerate(test_dataloader, 0):
        x0,_,_,file0,_ = data0
        file0 = file0[0]

        best = dict()
        for i, data1 in enumerate(train_dataloader, 0):
            _,x1,_,_,file1 = data1
            file1 = file1[0]

            # output1,output2 = net(Variable(x0).cuda(), Variable(x1).cuda()) #tracey
            output1,output2 = net(Variable(x0).cpu(), Variable(x1).cpu())
            euclidean_distance = F.pairwise_distance(output1, output2)
            distf = euclidean_distance.cpu().data.numpy()[0][0]
            print(".", end="", flush=True)
            #print("DIST: {:.2f}\t{}\t{}".format(distf, file0, file1))
            best[file1] = distf

            if (n < 10):
                concatenated = torch.cat((x0, x1), 0)
                imshow(torchvision.utils.make_grid(concatenated), 'DIST: {:.2f} {} {}'.format(distf, file0, file1))
                n += 1

        print(len(best))
        vsorted = sorted(best.items(), key=operator.itemgetter(1))
        for k,v in vsorted:
            print("{:.2f}\t{}\t{}".format(v, file0, k))






###################################################################################################
#  MAIN
###################################################################################################

vis_dataloader = DataLoader(SiameseNetworkDataset(Config.training_dir),
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)
dataiter = iter(vis_dataloader)


example_batch = next(dataiter)
concatenated = torch.cat((example_batch[0], example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated), 'example batch')
print(example_batch[2].numpy())


net = training()
testing(net)
