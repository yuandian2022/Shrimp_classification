# %%
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from inception_raman import *
import shap
import warnings

device = torch.device('cuda:0')

def seed_torch(seed=42):
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed(seed)
seed_torch()

# %%
class Mydata(Dataset):
    def __init__(self, data):
        self.datas = data
    
    def __getitem__(self, idx):
        one_data = self.datas[idx]
        name = one_data[0]
        label = int(one_data[1])
        spectrum = one_data[5:].astype(float)
        spectrum = torch.as_tensor(spectrum, dtype=torch.float)
        label = torch.as_tensor(label)
        return name, spectrum, label
    
    def __len__(self):
        return len(self.datas)

# %%
model = InceptionTime(2048) 
model.to(device)
model.load_state_dict(torch.load('inceptiontime.pt', map_location='cuda:0'))
loss_fn = nn.CrossEntropyLoss()

df = pd.read_csv('dataset_19504.csv')
df = np.array(df)

test_data_t = Mydata(df)
test_dataloader_t = DataLoader(test_data_t, batch_size=1)

# %%
input_list_t = []
for i in test_data_t:
    z = i[1].unsqueeze(0)
    input_list_t.append(np.array(z))
input_list_t = np.array(input_list_t)

random_indices = np.random.choice(input_list_t.shape[0], 200, replace=False)
input_list = input_list_t[random_indices]
input_list = torch.from_numpy(input_list)

remaining_indices = np.delete(np.arange(input_list_t.shape[0]), random_indices)
input_list_2 = input_list_t[remaining_indices]
input_list_2 = torch.from_numpy(input_list_2)

input_list_2 = input_list_2.to(device)
input_list = input_list.to(device)

# %%
warnings.filterwarnings('ignore')
print('DeepExplainer')
model.eval()
explainer = shap.DeepExplainer(model, input_list)
print('shap_values')
shap_values = explainer.shap_values(input_list_2)
print('finish')

shap_values_ = np.array(shap_values)
print(shap_values_.sum())
np.save('shap_value.npy',shap_values_)