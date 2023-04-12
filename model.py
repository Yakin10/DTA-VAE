import torch
import torch.nn as nn
from torch.autograd import Variable
    
class selfattention(nn.Module):
    def __init__(self, in_channels):
        super(selfattention, self).__init__()
        self.in_channels = in_channels

        self.w_query = nn.Linear(in_features=in_channels, out_features=in_channels, bias=False)
        self.w_key = nn.Linear(in_features=in_channels, out_features=in_channels, bias=False)
        self.w_value = nn.Linear(in_features=in_channels, out_features=in_channels, bias=False)

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, input):
        q = self.w_query(input)
        k = self.w_key(input)
        v = self.w_value(input)

        attention_matrix = torch.matmul(q, k.transpose(0, 1))
        attention_matrix = self.softmax(attention_matrix / (self.in_channels ** 0.5))

        out = torch.matmul(attention_matrix, v)
        return out


class Convo(nn.Module):
    def __init__(self, num_filters, k_size):
        super(Convo, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_filters*2,kernel_size=k_size, stride=1, padding=k_size//2),
            
        )
        self.conv2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters, num_filters * 4, k_size, 1, k_size//2),
            
        )
        self.conv3 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Conv1d(num_filters * 2, num_filters * 6, k_size, 1, k_size//2),
            
        )

        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=128, out_channels=num_filters, kernel_size=k_size, stride=1, padding=k_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(num_features=num_filters),
            nn.Conv1d(in_channels=num_filters, out_channels=num_filters*2, kernel_size=k_size, stride=1, padding=k_size//2),
            nn.ReLU(),
            nn.Conv1d(in_channels=num_filters*2, out_channels=num_filters*3, kernel_size=k_size, stride=1, padding=k_size//2)
        )
        self.out = nn.AdaptiveAvgPool1d(1)
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Linear(num_filters * 3, num_filters * 3),
            nn.ReLU()
        )

    def reparametrize(self, mean, logvar):
        std = logvar.mul(0.5).exp_()
        eps = torch.FloatTensor(std.size()).normal_(0,0.1)
        eps = Variable(eps)
        return eps.mul(std).add_(mean)

    def forward(self, x):
        x = self.conv1(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv2(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        x = self.conv3(x)
        out, gate = x.split(int(x.size(1) / 2), 1)
        x = out * torch.sigmoid(gate)
        output = self.out(x)
        output = output.squeeze()
        output1 = self.layer1(output)
        output2 = self.layer2(output)
        output = self.reparametrize(output1, output2)
        return output, output1, output2

class Convo1(nn.Module):
    def __init__(self, num_filters, kernel):
        super(Convo1, self).__init__()     
        self.linear = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128,num_filters*3)
        )
            
    def forward(self, x):
        output = self.linear(x)
        return output


class deconvo(nn.Module):
    def __init__(self, num_filters, size, kernel):
        super(deconvo, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(num_filters*3, num_filters*3*(80-3*(kernel-1))),
            nn.ReLU()
        )
        self.deconv1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=num_filters*3, out_channels=num_filters*2, kernel_size=kernel, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=num_filters*2, out_channels=num_filters, kernel_size=kernel, stride=1, padding=0),
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=num_filters, out_channels=128, kernel_size=kernel, stride=1, padding=0),
            nn.ReLU()
        )
        self.layer = nn.Linear(128, size)
    
    def forward(self, x, num_filters, kernel):
        x = self.layer1(x)
        x = self.deconv1(x.view(-1, num_filters*3, 80 - 3*(kernel-1)))
        x = self.layer(x.permute(0,2,1))
        return x

class net_r(nn.Module):
    def __init__(self, num_filters):
        super(net_r, self).__init__()
        
        
        self.reg = nn.Sequential(
            nn.Linear(num_filters*6,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1)
        )
        
        self.attention = selfattention(num_filters*3)
        
        self.reg1 = nn.Sequential(
            nn.Linear(num_filters*3, num_filters*3),
            nn.ReLU(),
        )
        
        self.reg2 = nn.Sequential(
            nn.Linear(num_filters*3, num_filters*3),
            nn.ReLU()
        )
    
    def forward(self, A, B):
        x = self.reg1(A)
        y = self.reg2(B)
        #x = self.attention(x)
        #y = self.attention(y)
        out = torch.cat((x,y),1)
        output = self.reg(out)
        return output
    

class net(nn.Module):
    def __init__(self, FLAGS, NUM_FILTERS, Kernel1, Kernel2):
        super(net, self).__init__()
        self.embedding1 = nn.Embedding(FLAGS.charsmiset_size, 128)
        self.cnn1 = Convo(NUM_FILTERS, Kernel1)
        self.cnn2 = Convo1(NUM_FILTERS, Kernel2)
        self.reg = net_r(NUM_FILTERS)
        self.decoder1 = deconvo(NUM_FILTERS, FLAGS.charsmiset_size, Kernel1)

    def forward(self, x, y, num_filters, kernel1):
        x_init = Variable(x.long())
        x = self.embedding1(x_init)
        x_embedding = x.permute(0, 2, 1)
        x, mu_x, logvar_x = self.cnn1(x_embedding)
        y = self.cnn2(y)
        out = self.reg(x, y).squeeze()
        x = self.decoder1(x, num_filters, kernel1)
        return out, x, x_init, mu_x, logvar_x,