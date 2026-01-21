# models.py
import torch
import torch.nn as nn
from torch.nn.init import xavier_normal_
from torch.nn import Parameter
import torch.nn.functional as F

class Discriminator(nn.Module):
    def __init__(self, in_size):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(in_size, 128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
    
    def forward(self, z):
        return self.model(z)

class PTP(nn.Module):
    def __init__(self, input_dim, rank, output_dim=60, poly_order=2):
        super(PTP, self).__init__()
        self.input_dim = input_dim
        self.rank = rank
        self.poly_order = poly_order
        self.output_dim = output_dim
        
        self.expanded_dim = input_dim + 1
        self.poly_dim = self.expanded_dim * poly_order
        
        self.factors = nn.ParameterList([
            nn.Parameter(torch.Tensor(self.poly_dim, rank))
            for _ in range(3)
        ])
        self.fusion_weights = nn.Parameter(torch.Tensor(rank, output_dim))
        self.fusion_bias = nn.Parameter(torch.Tensor(output_dim))
        
        for factor in self.factors:
            xavier_normal_(factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, audio_x, video_x, text_x):
        batch_size = audio_x.shape[0]
        device = audio_x.device
        
        def _expand(x):
            x = torch.cat([torch.ones(batch_size, 1).to(device), x], dim=1)
            return torch.cat([x ** (i+1) for i in range(self.poly_order)], dim=1)
        
        audio_poly = _expand(audio_x)
        video_poly = _expand(video_x)
        text_poly = _expand(text_x)
        
        fusion_audio = torch.matmul(audio_poly, self.factors[0])
        fusion_video = torch.matmul(video_poly, self.factors[1])
        fusion_text = torch.matmul(text_poly, self.factors[2])
        
        fusion_zy = fusion_audio * fusion_video * fusion_text
        output = torch.matmul(fusion_zy, self.fusion_weights) + self.fusion_bias
        
        return output

class MisaPTPGatedRec(nn.Module):
    def __init__(self, in_size, output_dim, hidden_size=20, hidden_size1=80, dropout=0.1):
        super(MisaPTPGatedRec, self).__init__()
      
        self.common = nn.Sequential(nn.Linear(in_size, 320), nn.ReLU(),
                                    nn.Linear(320, 128), nn.ReLU(),
                                    nn.Linear(128, 60), nn.ReLU())
        self.ptp = PTP(input_dim=60, rank=16, output_dim=60, poly_order=2)
        self.unique1 = nn.Sequential(nn.Linear(in_size, 320), nn.ReLU(),
                                     nn.Linear(320, 60), nn.ReLU())
        self.unique2 = nn.Sequential(nn.Linear(in_size, 256), nn.ReLU(),
                                     nn.Linear(256, 128), nn.ReLU(),
                                     nn.Linear(128, 60), nn.ReLU())
        self.unique3 = nn.Sequential(nn.Linear(in_size, 260), nn.ReLU(),
                                     nn.Linear(260, 130), nn.ReLU(),
                                     nn.Linear(130, 60), nn.ReLU())
       
        encoder1 = nn.Sequential(nn.Linear(60 * 4, 900), nn.ReLU(), nn.Dropout(p=dropout))
        encoder2 = nn.Sequential(nn.Linear(900, 512), nn.ReLU(), nn.Dropout(p=dropout))
        encoder3 = nn.Sequential(nn.Linear(512, 64), nn.ReLU(), nn.Dropout(p=dropout))
        encoder4 = nn.Sequential(nn.Linear(64, 15), nn.ReLU(), nn.Dropout(p=dropout))
        self.encoder = nn.Sequential(encoder1, encoder2, encoder3, encoder4)
        self.classifier = nn.Sequential(nn.Linear(15, output_dim), nn.Sigmoid())
        self.output_range = Parameter(torch.FloatTensor([8]), requires_grad=False)
        self.output_shift = Parameter(torch.FloatTensor([-4]), requires_grad=False)

        self.linear_h1 = nn.Sequential(nn.Linear(60, 60), nn.ReLU())
        self.linear_z1 = nn.Bilinear(60, 120, 60)
        self.linear_o1 = nn.Sequential(nn.Linear(60, 60), nn.ReLU(), nn.Dropout(p=dropout))

        self.linear_h2 = nn.Sequential(nn.Linear(60, 60), nn.ReLU())
        self.linear_z2 = nn.Bilinear(60, 120, 60)
        self.linear_o2 = nn.Sequential(nn.Linear(60, 60), nn.ReLU(), nn.Dropout(p=dropout))

        self.linear_h3 = nn.Sequential(nn.Linear(60, 60), nn.ReLU())
        self.linear_z3 = nn.Bilinear(60, 120, 60)
        self.linear_o3 = nn.Sequential(nn.Linear(60, 60), nn.ReLU(), nn.Dropout(p=dropout))

        self.rec1 = nn.Sequential(nn.Linear(120, 256), nn.ReLU(),
                                  nn.Linear(256, hidden_size1), nn.ReLU())
        self.rec2 = nn.Sequential(nn.Linear(120, 256), nn.ReLU(),
                                  nn.Linear(256, hidden_size1), nn.ReLU())
        self.rec3 = nn.Sequential(nn.Linear(120, 256), nn.ReLU(),
                                  nn.Linear(256, hidden_size1), nn.ReLU())

    def forward(self, x_gene, x_path, x_can):
        x_gene_common = self.common(x_gene)
        x_path_common = self.common(x_path)
        x_can_common = self.common(x_can)

        h1 = self.linear_h1(x_gene_common)
        vec31 = torch.cat((x_path_common, x_can_common), dim=1)
        z1 = self.linear_z1(x_gene_common, vec31)
        o1 = self.linear_o1(nn.Sigmoid()(z1) * h1)

        h2 = self.linear_h1(x_path_common)
        vec32 = torch.cat((x_gene_common, x_can_common), dim=1)
        z2 = self.linear_z1(x_path_common, vec32)
        o2 = self.linear_o1(nn.Sigmoid()(z2) * h2)

        h3 = self.linear_h1(x_can_common)
        vec33 = torch.cat((x_gene_common, x_path_common), dim=1)
        z3 = self.linear_z1(x_path_common, vec33)
        o3 = self.linear_o1(nn.Sigmoid()(z3) * h3)
        
        ptp = self.ptp(o1, o2, o3)
        
        x_gene_unique = self.unique1(x_gene)
        x_path_unique = self.unique2(x_path)
        x_can_unique = self.unique3(x_can)
      
        out_fusion = torch.cat((ptp, x_gene_unique, x_path_unique, x_can_unique), dim=1)
        encoder = self.encoder(out_fusion)
        out = self.classifier(encoder)
        out = out * self.output_range + self.output_shift

        gene_rec = self.rec1(torch.cat((ptp, x_gene_unique), dim=1))
        path_rec = self.rec2(torch.cat((ptp, x_path_unique), dim=1))
        can_rec = self.rec3(torch.cat((ptp, x_can_unique), dim=1))
        
        return out, x_gene, gene_rec, x_path, path_rec, x_can, can_rec, \
               x_gene_common, x_path_common, x_can_common, \
               x_gene_unique, x_path_unique, x_can_unique, ptp