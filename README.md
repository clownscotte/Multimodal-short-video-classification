# Multimodal-short-video-classification

data_extractor.py：实现数据预处理，video_to_audio将视频转码为音频，extract_video_Name提取视频的文件名，extract_video_Frames提取视频的帧（这里只取了128帧），输入均是文件名filename。

data.py：实现视频文件和标签文件的读取，这里考虑到视频顺序和标签顺序不一致，需要对其进行排序，以实现对应。

inceptionresnetv2.py实现了inceptionresnetv2网络，去掉最后的全连接层，输入为torch.Size([1, 3, 299, 299])，输出为torch.Size([1, 1536, 8, 8])。问题：不知道截取到哪一层，就只将最后一层全连接去掉。（1为batch_size，inceptionresnetv2模型下载地址：链接：https://pan.baidu.com/s/1jbREy49-wku1xYIYZJUkuQ 提取码：rva7）

修改：增加原有的池化层，输出[1536, 8]

netvlad.py将inceptionresnetv2.py输出特征图作为输入，输出num_clusters * dims维特征向量，也即64 * 1536（上面输出的通道数），然后接上一个全连接层输出维度为1024，最后输出为torch.Size([1, 1024])。（1为batch_size）

bert.py有一个函数get_tokens和一个TextNet类。get_token函数输入为经过增加[CLS]和[SEP]的纯文本以及bert预训练产生的tokenizer，输出为tokens、segments和input_masks作为文本网络的输入。TextNet有个参数code_length作为最终网络将文本生成的特征向量长度，这里定义为1024。最后输出维度为torch.Size([2, 1024])。（2表示输入文本为两句，bert模型下载地址：链接：https://pan.baidu.com/s/1yE09dpUh0NmaoqYoQLV4eg 提取码：hmg8）

lmf.py中LMF网络将图像和文本模态进行融合，输入图像（三维张量表示）、文本（纯文本形式加[CLS]和[SEP]）和tokenizer，输出维度为torch.Size([1, 1024])。（1表示batch_size）问题：output_dim和rank的选取。

修改：（1）加上语音信息，问题：不确定是否正确，仿照原有文件增加的（第26，32，51，54行）；（2）因为vggish是直接从github上导入，所以这个模型直接加载在了lmf文件中，没有另写（第24，49行）；（3）由于是一串帧信息被处理，所以不是直接inceptionresnetv2，修改为对一系列帧提取特征图，然后转化格式为([1539,8], 128])（第39-44行）

hmc.py定义了一个HMC模块和相关的loss。HMC参数为feature_size=1024（前面lmf得到的特征向量）以及其他参数根据分类数目进行定义，输入为lmf的输出，输出为得到的L1和L2标签。hmc_loss计算损失函数。

model.py整合前面网络结构，参数为一级标签个数、二级标签个数以及L12_table，输入为图像和文本以及tokenizer，输出为一级标签和二级标签相应位置概率。

修改：增加数据处理（第28-30行）。
