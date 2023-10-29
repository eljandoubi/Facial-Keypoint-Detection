import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


def init_weights(m: nn)->None:
    
    if isinstance(m, nn.Conv2d):
        I.uniform_(m.weight)
        
    elif isinstance(m, nn.Linear):
        I.xavier_uniform_(m.weight)
        
def out_size(l: list, s: int = 224)-> int:
    for e in l:
        s= (s-e+1)//2
    return s**2
        

class Flatten(nn.Module):
    def __init__(self)-> None:
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.size(0), -1)

        

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int = 1, out_channels: int = 3,
                 kernel_size: int = 3, p: float = 0.1)-> None:
            
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=0),
            nn.ELU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=p),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)
        
class LinBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int,
                 use_elu: bool = True, p: float = 0.1)-> None:
        
        super().__init__()
        
        if p==0:
            self.lin = nn.Linear(in_channels,out_channels)
        
        elif use_elu:
            self.lin = nn.Sequential(
                nn.Linear(in_channels,out_channels),
                nn.ELU(),
                nn.Dropout(p=p),
            )

        else:
            self.lin = nn.Sequential(
                nn.Linear(in_channels,out_channels),
                nn.Dropout(p=p),
            )
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x)



class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        
        c = 8
        d = 2
        channels = [1,]+[2**i for i in range(5,c+1)]
        filters = list(range(c-4,0,-1))
        denses = [out_size(filters)*2**c]+[1000,]*d+[2*68]
        ps = [(0.6*i)/(c+d-4) for i in range(1,c+d-3)]+[0.]
        
        self.net = nn.Sequential()
        
        for i in range(len(channels)-1):
            self.net.add_module(f"ConvBlock_{i}",
                                ConvBlock(channels[i], channels[i+1],
                                          filters[i], ps[i] 
                                         )
                               )
                                
        self.net.add_module("Flatten",Flatten())            
                                
        for j in range(len(denses)-1):
            self.net.add_module(f"LinBlock_{j}",
                                LinBlock(denses[j], denses[j+1],
                                          j==0, ps[i+j+1] 
                                         )
                               )
                                
        self.apply(init_weights)                 

        
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    m = Net()
    
    print(m)
    
    x=torch.randn(1,1,224,224)
    
    print(m(x))