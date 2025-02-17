import torch
import torch.nn as nn
import torch.nn.functional as F

def correct_sizes(sizes):
    corrected_sizes = [s if s % 2 != 0 else s - 1 for s in sizes]
    return corrected_sizes

def pass_through(x):
    return x

class Inception(nn.Module):
    def __init__(self, in_channels, n_filters, kernel_sizes=[9, 19, 39], bottleneck_channels=32, activation=nn.ReLU(),
                 return_indices=False):

        super(Inception, self).__init__()
        self.return_indices = return_indices

        if in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels=in_channels,
                                        out_channels=bottleneck_channels,
                                        kernel_size=1,
                                        stride=1,
                                        bias=False
                                        )
        else:
            self.bottleneck = pass_through
            bottleneck_channels = 1

        self.conv_from_bottleneck_1 = nn.Conv1d(in_channels=bottleneck_channels,
                                                out_channels=n_filters,
                                                kernel_size=kernel_sizes[0],
                                                stride=1,
                                                padding=kernel_sizes[0] // 2,
                                                bias=False
                                                )

        self.conv_from_bottleneck_2 = nn.Conv1d(in_channels=bottleneck_channels,
                                                out_channels=n_filters,
                                                kernel_size=kernel_sizes[1],
                                                stride=1,
                                                padding=kernel_sizes[1] // 2,
                                                bias=False
                                                )

        self.conv_from_bottleneck_3 = nn.Conv1d(in_channels=bottleneck_channels,
                                                out_channels=n_filters,
                                                kernel_size=kernel_sizes[2],
                                                stride=1,
                                                padding=kernel_sizes[2] // 2,
                                                bias=False)

        self.max_pool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1, return_indices=return_indices)
        self.conv_from_maxpool = nn.Conv1d(in_channels=in_channels,
                                           out_channels=n_filters,
                                           kernel_size=1,
                                           stride=1,
                                           padding=0,
                                           bias=False)

        self.batch_norm = nn.BatchNorm1d(num_features=4 * n_filters)
        self.activation = activation

    def forward(self, x):
        z_bottleneck = self.bottleneck(x)
        if self.return_indices:
            z_maxpool, indices = self.max_pool(x)
        else:
            z_maxpool = self.max_pool(x)

        z1 = self.conv_from_bottleneck_1(z_bottleneck)
        z2 = self.conv_from_bottleneck_2(z_bottleneck)
        z3 = self.conv_from_bottleneck_3(z_bottleneck)
        z4 = self.conv_from_maxpool(z_maxpool)

        z = torch.cat([z1, z2, z3, z4], axis=1)
        z = self.activation(self.batch_norm(z))
        if self.return_indices:
            return z, indices
        else:
            return z

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, n_filters=32, kernel_sizes=[9, 19, 39], bottleneck_channels=32, use_residual=True,
                 activation=nn.ReLU(), return_indices=False):
        super(InceptionBlock, self).__init__()
        self.use_residual = use_residual
        self.return_indices = return_indices
        self.activation = activation
        self.inception_1 = Inception(in_channels=in_channels,
                                     n_filters=n_filters,
                                     kernel_sizes=kernel_sizes,
                                     bottleneck_channels=bottleneck_channels,
                                     activation=activation,
                                     return_indices=return_indices
                                     )

        self.inception_2 = Inception(in_channels=4 * n_filters,
                                     n_filters=n_filters,
                                     kernel_sizes=kernel_sizes,
                                     bottleneck_channels=bottleneck_channels,
                                     activation=activation,
                                     return_indices=return_indices
                                     )

        self.inception_3 = Inception(in_channels=4 * n_filters,
                                     n_filters=n_filters,
                                     kernel_sizes=kernel_sizes,
                                     bottleneck_channels=bottleneck_channels,
                                     activation=activation,
                                     return_indices=return_indices
                                     )

        if use_residual:
            self.residual = nn.Sequential(nn.Conv1d(in_channels=in_channels,
                                                    out_channels=4 * n_filters,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0
                                                    ),
                                          nn.BatchNorm1d(num_features=4 * n_filters)
                                          )

    def forward(self, x):
        if self.return_indices:
            z, i1 = self.inception_1(x)
            z, i2 = self.inception_2(z)
            z, i3 = self.inception_3(z)
        else:
            z = self.inception_1(x)
            z = self.inception_2(z)
            z = self.inception_3(z)

        if self.use_residual:
            z = z + self.residual(x)
            z = self.activation(z)

        if self.return_indices:
            return z, [i1, i2, i3]
        else:
            return z

class Flatten(nn.Module):
    def __init__(self, out_features):
        super(Flatten, self).__init__()
        self.output_dim = out_features

    def forward(self, X):
        return X.view(-1, self.output_dim)


class Reshape(nn.Module):
    def __init__(self, out_shape):
        super(Reshape, self).__init__()
        self.out_shape = out_shape

    def forward(self, X):
        return X.view(-1, *self.out_shape)

class InceptionTime_blk1(nn.Module):
    def __init__(self, input_dim):
        super(InceptionTime_blk1, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, input_dim)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        return output
    
class InceptionTime_blk2(nn.Module):
    def __init__(self, input_dim):
        super(InceptionTime_blk2, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, input_dim)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        return output

class InceptionTime_blk3(nn.Module):
    def __init__(self, input_dim):
        super(InceptionTime_blk3, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, input_dim)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        return output
    
class GAM_Attention(nn.Module):  
    def __init__(self, in_channels, out_channels, rate=4):  
        super(GAM_Attention, self).__init__()  

        self.channel_attention = nn.Sequential(  
            nn.Linear(in_channels, int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Linear(int(in_channels / rate), in_channels)  
        )  
      
        self.spatial_attention = nn.Sequential(  
            nn.Conv1d(in_channels, int(in_channels / rate), kernel_size=3, padding=2),  
            nn.BatchNorm1d(int(in_channels / rate)),  
            nn.ReLU(inplace=True),  
            nn.Conv1d(int(in_channels / rate), out_channels, kernel_size=3, padding=0),  
            nn.BatchNorm1d(out_channels)  
        )  
      
    def forward(self, x):  
        b, c, _ = x.shape
        x_permute = x.permute(0, 2, 1)
        x_att_permute = self.channel_attention(x_permute)
        x_channel_att = x_att_permute.permute(0, 2, 1) 
        x = x * x_channel_att
        x_spatial_att = self.spatial_attention(x).sigmoid()  
        out = x * x_spatial_att

        return out  
    
class InceptionTime_GAM(nn.Module):
    def __init__(self, input_dim):
        super(InceptionTime_GAM, self).__init__()
        self.InceptionTime_Layers_1 = nn.Sequential(
            Reshape(out_shape=(1, input_dim)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            )
        )
        self.GAM_attention = GAM_Attention(128, 128)

        self.InceptionTime_Layers_2 = nn.Sequential(
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            )
        )
        
        self.InceptionTime_Layers_3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        x_block1 = self.InceptionTime_Layers_1(x)
        x_block2 = self.InceptionTime_Layers_2(x_block1)
        x_block2 = self.GAM_attention(x_block2)
        x_block = torch.cat([x_block1, x_block2], axis=1)
        output = self.InceptionTime_Layers_3(x_block)
        return output
    
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)
    
class InceptionTime_SEnet6(nn.Module):
    def __init__(self, input_dim):
        super(InceptionTime_blk2(), self).__init__()
        self.InceptionTime_Layers_1 = nn.Sequential(
            Reshape(out_shape=(1, input_dim)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
        )
        
        self.InceptionTime_Layers_2 = nn.Sequential(
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
        )
        
        self.InceptionTime_Layers_3 = nn.Sequential(
            nn.AdaptiveAvgPool1d(output_size=1),
            nn.Flatten(),
            nn.Linear(in_features=768, out_features=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        z1_b1, z2_b1, z3_b1 = self.InceptionTime_Layers_1(x)
        z1_b2, z2_b2, z3_b2 = self.InceptionTime_Layers_2(z3_b1)
        x_block = torch.cat([z1_b1, z2_b1, z3_b1, z1_b2, z2_b2, z3_b2], axis=1)
        output = self.InceptionTime_Layers_3(x_block)
        return output


class InceptionTime_SEnet6_lstm(nn.Module):
    def __init__(self, input_dim):
        super(InceptionTime_SEnet6_lstm, self).__init__()
        self.InceptionTime_Layers_1 = nn.Sequential(
            Reshape(out_shape=(1, input_dim)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
        )
        
        self.InceptionTime_Layers_2 = nn.Sequential(
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
        )
        
        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size=1024)
        self.lstm = nn.LSTM(1024, 256, 2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        z1_b1, z2_b1, z3_b1 = self.InceptionTime_Layers_1(x)
        z1_b2, z2_b2, z3_b2 = self.InceptionTime_Layers_2(z3_b1)
        x_block = torch.cat([z1_b1, z2_b1, z3_b1, z1_b2, z2_b2, z3_b2], axis=1)
        x_block = self.avgpool1d(x_block)
        out, (h1, c1) = self.lstm(x_block)
        out = out[:, -1, :]
        out = self.linear(out)
        output = self.softmax(out)
        return output
    
class InceptionTime_add_lstm(nn.Module):
    def __init__(self, input_dim):
        super(InceptionTime_add_lstm, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, input_dim)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            )
        )

        self.lstm = nn.LSTM(2048, 256, 2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.InceptionTime_Layers(x)
        # x = self.avgpool1d(x)
        out, (h1, c1) = self.lstm(x)
        out = out[:, -1, :]
        out = self.linear(out)
        output = self.softmax(out)
        return output
    
class InceptionTime_cat_lstm(nn.Module):
    def __init__(self, input_dim):
        super(InceptionTime_cat_lstm, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, input_dim)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1)
        )
        
        self.lstm = nn.LSTM(input_size=16, hidden_size=32 * 4, num_layers=1, batch_first=True)
        self.flatten = Flatten(out_features=256)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.Linear = nn.Linear(in_features=256, out_features=2)
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x_1 = [x[:, :, i:i + 16] for i in range(0, 2048, 16)]
        x_1 = torch.stack(x_1, dim=1).squeeze(2)# (batch_size, sequence_length, d_model)
        # x_1 = self.linear_layer_1(x_1)
        
        x_1,(h,c) = self.lstm(x_1)
        x_1 = x_1[:,-1,:] 
        x_2 = self.InceptionTime_Layers(x).squeeze(dim=2)
        output = torch.cat((x_1, x_2), dim=1)
        
        output = self.flatten(output)
        output = self.relu(output)
        output = self.dropout(output)
        output = self.Linear(output)
        output = self.softmax(output)
        return output
    
class InceptionTime_add_GRU(nn.Module):
    def __init__(self, input_dim):
        super(InceptionTime_add_GRU, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, input_dim)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[5, 11, 23],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            )
        )

        self.avgpool1d = nn.AdaptiveAvgPool1d(output_size=2048)
        self.gru = nn.GRU(2048, 256, 2, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(512, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.InceptionTime_Layers(x)
        x = self.avgpool1d(x)
        out, h1 = self.gru(x)
        out = out[:, -1, :]
        out = self.linear(out)
        output = self.softmax(out)
        return output
