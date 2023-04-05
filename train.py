import torch
import numpy as np
from net import Net
from tqdm import tqdm
from data import MyDataset
from torch import nn, optim
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# fix random seed
seed = 1
torch.manual_seed(seed)            # torch的CPU随机性，为CPU设置随机种子
torch.cuda.manual_seed_all(seed)   # torch的GPU随机性，为所有GPU设置随机种子
np.random.seed(seed)


# data
root = 'E:/PIV_DATA2'
data_loader = DataLoader(MyDataset(root), batch_size=10, shuffle=True)

root2 = 'E:/PIV_DATA3'
data_loader2 = DataLoader(MyDataset(root2), batch_size=10, shuffle=True)


# net
net = Net().to(device)
opt = optim.Adam(net.parameters(), lr=1e-4)
loss_fun = nn.CrossEntropyLoss().to(device)
# net.load_state_dict(torch.load('model.param'))


# train
loss_record = []
loss_record2 = []
num_epochs = 500

for epoch in range(num_epochs):

    net.train()
    loop = tqdm(enumerate(data_loader), total=len(data_loader))
    epoch_loss = []
    for i, (image, mask) in loop:
        image, mask = image.to(device), mask.to(device)
        out_mask = net(image)
        loss = loss_fun(out_mask, mask.long())
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss.append(loss.item())
        loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
        loop.set_postfix(loss=loss.item())
    loss_record.append(np.mean(epoch_loss))
    
    
    net.eval()
    loop = tqdm(enumerate(data_loader2), total=len(data_loader2))
    epoch_loss = []
    with torch.no_grad():
        for i, (image, mask) in loop:
            image, mask = image.to(device), mask.to(device)
            out_mask = net(image)
            loss = loss_fun(out_mask, mask.long())
            epoch_loss.append(loss.item())
            loop.set_description(f'Epoch [{epoch}/{num_epochs}]')
            loop.set_postfix(loss=loss.item())
    loss_record2.append(np.mean(epoch_loss))


    if (epoch+1) % 50 == 0:
        torch.save(net.state_dict(), 'model.param')
        np.save('loss_record.npy', loss_record)
        np.save('loss_record2.npy', loss_record2)


# save
np.save('loss_record.npy', loss_record)
np.save('loss_record2.npy', loss_record2)
