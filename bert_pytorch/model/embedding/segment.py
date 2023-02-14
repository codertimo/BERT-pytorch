import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        """
        和TokenEmbedding不同，下面的__init__()中的第一个参数不是vocab_size而是3，
        因为SegmentEmbedding实例化后的input是segment_info，一个由0,1,2三种元素组成的向量
        故SegmentEmbedding初始化时只需要初始化3个向量即可；
        """
        super().__init__(3, embed_size, padding_idx=0)
