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
        """

        :param in_channels:             Number of input channels (input features)
        :param n_filters:               Number of filters per convolution layer => out_channels = 4*n_filters
        :param kernel_sizes:            List of kernel sizes for each convolution.
                                            Each kernel size must be odd number that meets -> "kernel_size % 2 != 0"
                                            This is neccessary bacause of padding size
                                            for correction of kernel_sizes use function "correct_sizes"
        :param bottleneck_channels:     Number of output channels in bottleneck.
                                            bottlemneck wont be used if the input_channel is 1
        :param activation:              Activation function for output tensor(nn.Relu())
        :param return_indices:          indices are needed only if we want to creat decoder with Inception with MaxUnpool1d
        """
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


class InceptionTime(nn.Module):
    def __init__(self, input_dim):
        super(InceptionTime, self).__init__()
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


class InceptionTimeBinary(nn.Module):
    def __init__(self):
        super(InceptionTimeBinary, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 1600)),
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


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(160, 500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU()
        )


class InceptionTimeFE(nn.Module):
    def __init__(self):
        super(InceptionTime, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 500)),
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
            nn.Linear(in_features=32 * 4 * 1, out_features=32 * 4 * 1),
            nn.LayerNorm(32 * 4 * 1),

        )

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        return output


class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.mlpc = nn.Sequential(
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=23),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.mlpc(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, classes):
        super(Classifier, self).__init__()
        self.mlpc = nn.Sequential(
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=classes),
            nn.ReLU(),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        output = self.mlpc(x)
        return output


class GatedGAN(nn.Module):
    def __init__(self, d_ft_sub: int = 128, d_ft: int = 256):
        super(GatedGAN, self).__init__()
        self.inceptiontime1 = InceptionTimeFE()
        self.inceptiontime2 = InceptionTimeFE()

        self.gate = torch.nn.Linear(d_ft_sub + d_ft_sub, 2)

        self.classifier = Classifier()
        self.discriminator = Discriminator()

    def forward(self, acc, gyro):
        ft_acc = self.inceptiontime1(acc)
        ft_gyro = self.inceptiontime2(gyro)

        gate = F.softmax(self.gate(torch.cat([ft_acc, ft_gyro], dim=-1)), dim=-1)

        ft = torch.cat([ft_acc * gate[:, 0:1], ft_gyro * gate[:, 1:2]], dim=-1)

        output_c = self.classifier(ft)
        output_d = self.discriminator(ft)

        return gate, ft, output_c, output_d


class InceptionTimeBinaryNoDomain(nn.Module):
    def __init__(self):
        super(InceptionTimeBinaryNoDomain, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 800)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[9, 19, 39],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[9, 19, 39],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[9, 19, 39],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[9, 19, 39],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(in_features=4 * 32 * 1, out_features=2),
            nn.Softmax(dim=-1)
        )

        # self.Classfier = Classifier(classes=23)

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        # output = self.Classfier(output)
        return output


class InceptionTimeBinaryDomain(nn.Module):
    def __init__(self):
        super(InceptionTimeBinaryNoDomain, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 800)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[9, 19, 39],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[9, 19, 39],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=2),
            Flatten(out_features=32 * 4 * 2),
            # nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            # nn.ReLU(),
            # nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            # nn.ReLU(),
            # nn.Linear(in_features=4 * 32 * 2, out_features=2),
            # nn.Softmax(dim=-1)
        )

        self.Classfier = Classifier(classes=2)
        self.Discriminator = Discriminator(classes=5)

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        output_c = self.Classfier(output)
        output_d = self.Discriminator(output)
        return output, output_c, output_d

class InceptionTimeMonitor_1blks(nn.Module):
    def __init__(self, ks1=7, ks2=15, ks3=31):
        super(InceptionTimeMonitor_1blks, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 800)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=23),
            nn.Softmax(dim=1)
        )

        # self.Classfier = Classifier(classes=23)

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        # output = self.Classfier(output)
        return output

class InceptionTimeMonitor_2blks(nn.Module):
    def __init__(self, ks1=7, ks2=15, ks3=31):
        super(InceptionTimeMonitor_2blks, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 800)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=23),
            nn.Softmax(dim=1)
        )

        # self.Classfier = Classifier(classes=23)

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        # output = self.Classfier(output)
        return output


class InceptionTimeMonitor_3blks(nn.Module):
    def __init__(self, ks1=7, ks2=15, ks3=31):
        super(InceptionTimeMonitor_3blks, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 800)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=23),
            nn.Softmax(dim=1)
        )

        # self.Classfier = Classifier(classes=23)

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        # output = self.Classfier(output)
        return output


class InceptionTimeMonitor_4blks(nn.Module):
    def __init__(self, ks1=7, ks2=15, ks3=31):
        super(InceptionTimeMonitor_4blks, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 800)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=23),
            nn.Softmax(dim=1)
        )

        # self.Classfier = Classifier(classes=23)

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        # output = self.Classfier(output)
        return output

class InceptionTimeMonitor_5blks(nn.Module):
    def __init__(self, ks1=7, ks2=15, ks3=31):
        super(InceptionTimeMonitor_5blks, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 800)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=23),
            nn.Softmax(dim=1)
        )

        # self.Classfier = Classifier(classes=23)

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        # output = self.Classfier(output)
        return output

class InceptionTimeMonitor_6blks(nn.Module):
    def __init__(self, ks1=7, ks2=15, ks3=31):
        super(InceptionTimeMonitor_6blks, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 800)),
            InceptionBlock(
                in_channels=1,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),
            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            InceptionBlock(
                in_channels=32 * 4,
                n_filters=32,
                kernel_sizes=[ks1, ks2, ks3],
                bottleneck_channels=32,
                use_residual=True,
                activation=nn.ReLU()
            ),

            nn.AdaptiveAvgPool1d(output_size=1),
            Flatten(out_features=32 * 4 * 1),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=4 * 32 * 1),
            nn.ReLU(),
            nn.Linear(in_features=4 * 32 * 1, out_features=23),
            nn.Softmax(dim=1)
        )

        # self.Classfier = Classifier(classes=23)

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        # output = self.Classfier(output)
        return output

class InceptionTimeMonitor_mlp(nn.Module):
    def __init__(self):
        super(InceptionTimeMonitor_mlp, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(800, 1600),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(1600, 500),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(500, 128),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.classifier = Classifier()

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        output_c = self.classifier(output)

        return output_c


class InceptionTimeMonitor_fcn(nn.Module):
    def __init__(self):
        super(InceptionTimeMonitor_fcn, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 800)),

            nn.Conv1d(in_channels=1, out_channels=128, kernel_size=7, stride=1, padding=7 // 2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(in_channels=128, out_channels=256, kernel_size=5, stride=1, padding=5 // 2),
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.Conv1d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=3 // 2),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(output_size=1),

            Flatten(out_features=32 * 4 * 1),
        )

        self.classifier = Classifier()

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        output_c = self.classifier(output)

        return output_c


class ResidualBlock(nn.Module):

    def __init__(self, in_maps, out_maps):
        super(ResidualBlock, self).__init__()
        self.in_maps = in_maps
        self.out_maps = out_maps

        self.reshape = Reshape(out_shape=(1, 160))

        self.conv1 = nn.Conv1d(self.in_maps, self.out_maps, kernel_size=7, stride=1, padding=7 // 2)
        self.bn1 = nn.BatchNorm1d(self.out_maps)
        self.ac1 = nn.ReLU()

        self.conv2 = nn.Conv1d(self.out_maps, self.out_maps, kernel_size=5, stride=1, padding=5 // 2)
        self.bn2 = nn.BatchNorm1d(self.out_maps)
        self.ac2 = nn.ReLU()

        self.conv3 = nn.Conv1d(self.out_maps, self.out_maps, kernel_size=3, stride=1, padding=3 // 2)
        self.bn3 = nn.BatchNorm1d(self.out_maps)
        self.ac3 = nn.ReLU()

    def forward(self, x):
        x = self.ac1(self.bn1(self.conv1(x)))
        inx = x
        x = self.ac2(self.bn2(self.conv2(x)))
        x = self.ac3(self.bn3(self.conv3(x) + inx))

        return x


class InceptionTimeMonitor_resnet(nn.Module):
    def __init__(self):
        super(InceptionTimeMonitor_resnet, self).__init__()
        self.InceptionTime_Layers = nn.Sequential(
            Reshape(out_shape=(1, 800)),

            # print("first residual block"),
            ResidualBlock(in_maps=1, out_maps=64),
            # print("second residual block"),
            ResidualBlock(in_maps=64, out_maps=128),
            # print("third residual block"),
            ResidualBlock(in_maps=128, out_maps=128),

            nn.AdaptiveAvgPool1d(output_size=1),

            Flatten(out_features=32 * 4 * 1)
        )

        self.classifier = Classifier()

    def forward(self, x):
        output = self.InceptionTime_Layers(x)
        output_c = self.classifier(output)

        return output_c

class InceptionTimeMonitor_lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(InceptionTimeMonitor_lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.InceptionTime_Layers = nn.Sequential(
            nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True)
        )

        self.fc = nn.Linear(hidden_size, 128)
        self.classifier = Classifier()

    def forward(self, x):
        output, (h_n, h_c) = self.InceptionTime_Layers(x)
        output = self.fc(output[:, -1, :])
        output_c = self.classifier(output)

        return output_c

class InceptionTimeMonitor_bilstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(InceptionTimeMonitor_bilstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.InceptionTime_Layers = nn.Sequential(
            nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        )

        self.fc = nn.Linear(2 * hidden_size, 128)
        self.classifier = Classifier()

    def forward(self, x):
        output, (h_n, h_c) = self.InceptionTime_Layers(x)

        out_forward = output[range(len(output)), 160 - 1, :self.hidden_size]
        out_reverse = output[:, 0, self.hidden_size:]
        out_reduced = torch.cat((out_forward, out_reverse), 1)

        output = self.fc(out_reduced)
        output_c = self.classifier(output)

        return output_c