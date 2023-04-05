
import cv2
import torch
import win32ui
import matplotlib.pyplot as plt
from net import Net

model = Net()
# model.load_state_dict(torch.load('model.param'))

dlg = win32ui.CreateFileDialog(1)
dlg.SetOFNInitialDir('')
dlg.DoModal()
Path = dlg.GetPathName() 

img = cv2.imread(Path, cv2.IMREAD_GRAYSCALE)
img_in = torch.tensor(img/img.max()).unsqueeze(dim=0)
img_in = img_in.unsqueeze(dim=0).float()
mask_out = torch.softmax(model(img_in)[0],dim=0)
mask_out = torch.argmax(mask_out, dim=0).numpy()*255

plt.figure()
ax1 = plt.subplot(1, 2, 1)
plt.imshow(img)
ax2 = plt.subplot(1, 2, 2)
plt.imshow(mask_out)






