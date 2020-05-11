import os
from PIL import Image
import xlrd
import torch.utils.data as data
import torchvision.transforms as transforms


class VideoDataset(data.Dataset):
    def __init__(self, video_root,label_file):
        #self.trainsize = trainsize
        self.video = [video_root + f for f in os.listdir(video_root) if f.endswith('.mp4')]
        self.video = sorted(self.video)
        self.gl = self.getLabel(label_file)

        self.size = len(self.video)
        #self.img_transform = transforms.Compose([
            #transforms.Resize((self.trainsize, self.trainsize)),
        # transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def getLabel(self, label_file):
        Label_sheet = xlrd.open_workbook(label_file)
        Label = Label_sheet.sheet_by_index(0)
        video_Label_list = []
        Labels=[]
        for each_row in range(Label.nrows):
            video_label = Label.row_values(each_row)
            #Label_dict[video_label[0]] = video_label[1]
            video_Label_list.append(video_label)

        # print(Label_list)
        video_Label_list_s = sorted(video_Label_list, key=lambda x: x[0])
        for Label_list in video_Label_list_s:
            Labels.append(Label_list[1])
        return Labels

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')


    def __len__(self):
        return self.size


def get_loader(video_root, label_file, batchsize,  shuffle=True, num_workers=12, pin_memory=True):

    dataset = VideoDataset(video_root, label_file)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader



"""
class SalientTest(data.Dataset):
    def __init__(self, image_root, label_root, image_size):
        super(SalientTest, self).__init__()
        self.image_size = image_size
        self.image_list = list(map(lambda x: os.path.join(image_root, x), os.listdir(image_root)))
        self.label_list = list(map(lambda x: os.path.join(label_root, x), os.listdir(label_root)))
        self.image_list = sorted(self.image_list)
        self.label_list = sorted(self.label_list)
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.label_transform = transforms.ToTensor()

    def __getitem__(self, index):
        image = Image.open(self.image_list[index]).convert('RGB')
        label = Image.open(self.label_list[index]).convert('L')
        image = self.image_transform(image).unsqueeze(0)
        label = self.label_transform(label)
        name = self.image_list[index].split('/')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        return image, label, name

    def __len__(self):
        return len(self.image_list)
"""


