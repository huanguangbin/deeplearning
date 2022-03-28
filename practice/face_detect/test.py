from torch.utils.data import DataLoader
from dataloader.wider_face import WiderFaceDataset
txt_path=r"D:\Dataset\widerface\data\wider_face_val_bbx_gt.txt"
imgs_path=r"D:\Dataset\widerface\data\images"
dataset = WiderFaceDataset(txt_path,imgs_path)
trainloader=DataLoader(dataset)
for batch, (batch_x, batch_y) in enumerate(trainloader):
    pass
pass