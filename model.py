import torch
from lmf import LMF
#from hmc import HMC
from one import classfication as cf
import torch.nn as nn
import data_extractor
import torch.nn.functional as F
import torch.optim
from pytorch_transformers import BertTokenizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE=2

class VideoClassificationModel(nn.Module):
    def __init__(self, L1_labels_num,):
        super(VideoClassificationModel, self).__init__()
        self.L1_labels_num = L1_labels_num
        self.lmf = LMF()
        #self.hmc = HMC(feature_size=1024, L1_labels_num=self.L1_labels_num, L2_labels_num=L2_labels_num, L12_table=self.L12_table)
        self.Label= cf(feature_size=1024, L1_labels_num=self.L1_labels_num)
        self.mls = nn.MSELoss()
        self.opt = torch.optim.Adam(params=self.parameters(), lr=0.001)


    def forward(self, images, texts, tokenizer):
        out_lmf = self.lmf(audio, images, texts, tokenizer)
        L1= self.Label(out_lmf)
        return L1



    def train(self, x, label):

        out = self.forward(x)
        loss = self.mls(out, label)

        self.opt.zero_grad()

        loss.backward()

        self.opt.step()

#def train(model, device, train_loader, optimizer, epoch):
    # model.train()
    #for batch_idx, (data, target) in enumerate(train_loader):
        # data, target = data.to(device), target.to(device)
        # optimizer.zero_grad()
        #output = model(data)
        # loss = F.nll_loss(output, target)
        # loss.backward()
        #optimizer.step()
        #print(batch_idx, loss.item())

if __name__ == '__main__':
    #images = torch.randn(1, 3, 299, 299)
    #texts = ['[CLS] 你是谁? 你是好人. [SEP]']
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    texts=data_extractor.extract_video_Name(filename)
    images=data_extractor.extract_video_Frames(filemane)
    audio=data_extractor.video_to_audio(filename)
    model=VideoClassificationModel().to(DEVICE)

    model = VideoClassificationModel(7)
    model_image = model.lmf.inceptionresnetv2
    pretrained_dict = torch.load('inceptionresnetv2-520b38e4.pth')
    model_dict = model_image.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(pretrained_dict)
    model_image.load_state_dict(model_dict)

    L1= model(images, texts, tokenizer)
    print(L1.shape)
    #print(L2.shape)
