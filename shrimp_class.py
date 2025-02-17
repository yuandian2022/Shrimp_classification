#%% 
### Load libraries
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, roc_auc_score, average_precision_score
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
from inception_raman import *
from Cmp_CNN import *
import xlsxwriter

device = torch.device('cuda:0')

# %% 
### Import data and normalize
dataset = pd.read_csv('dataset_19504.csv')  # '≤ 5': 0, '5~100': 1, '100~500': 2, '500~1000': 3, '≥ 1000': 4
# 选择dataframe中第4列内容为1和5的行
model_data = dataset[dataset['Group'].isin([0, 1])]

df_normalize = model_data.copy()
columns_to_normalize = df_normalize.columns[5:]
f = lambda x: (x - x.min()) / (x.max() -x.min())
df_normalize[columns_to_normalize] = df_normalize[columns_to_normalize].apply(f, axis=1)

#%%
### Randomly select 20% of the shrimp data as the test set
random_seed = 42
random.seed(random_seed)
group_names = df_normalize['Shrimp_id'].unique().tolist()
num_selected = int(len(group_names) * 0.2)
selected_data = random.sample(group_names, num_selected)

test_df = df_normalize[df_normalize['Shrimp_id'].isin(selected_data)]
train_df = df_normalize[~df_normalize['Shrimp_id'].isin(selected_data)]

test_data = np.array(test_df)
train_data = np.array(train_df)

### Extracting features and labels
x_train = train_df.iloc[:, 5:].values.reshape(-1, 2048)
y_train = train_df.iloc[:, 1:2].values.reshape(-1, 1).astype('int')
x_test = test_df.iloc[:, 5:].values.reshape(-1, 2048)
y_test = test_df.iloc[:, 1:2].values.reshape(-1, 1).astype('int')

#%%
##############################################################################
### Train and evaluate the random forest model
rf_model = RandomForestClassifier(n_estimators=100)
rf_model.fit(x_train, y_train.ravel())
y_pred_rf = rf_model.predict(x_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"RF模型准确率: {accuracy_rf:.4f}")

### Train and evaluate the SVM model
svm_model = SVC()  
svm_model.fit(x_train, y_train.ravel())
y_pred_svm = svm_model.predict(x_test)
accuracy_svm = accuracy_score(y_test, y_pred_svm)
print(f"SVM模型准确率: {accuracy_svm:.4f}")

### Train and evaluate the XGBoost model
xgb_model = XGBClassifier()  
xgb_model.fit(x_train, y_train.ravel()) 
y_pred_xgb = xgb_model.predict(x_test) 
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost模型准确率: {accuracy_xgb:.4f}")

# %%
##############################################################################
### Train and evaluate the InceptionTime model

# Data loading
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

writer = SummaryWriter('./tensorboard')

# Define five-fold cross validation
num_folds = 2
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)

model_acc = []
model_list = []

for fold, (train_index, val_index) in enumerate(kf.split(train_data)):
    print(f'Start fold {fold+1}')
    train_fold = train_data[train_index]
    val_fold = train_data[val_index]

    train_dataset = Mydata(train_fold)
    valid_dataset = Mydata(val_fold)
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=32, shuffle=True)

    # Creating and training the model
    model = InceptionTime(2048)
    model.to(device)
    loss_fn = nn.CrossEntropyLoss()
    loss_fn.to(device)
    learn_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.1)
    
    best_valid_accuracy = 0
    wait = 0
    
    epoch = 50
    for i in range(epoch):
        total_train_loss = 0
        total_train_accuracy = 0

        model.train()
        for name, spectrum, label in train_dataloader:
            spectrum = spectrum.to(device)
            label = label.to(device)
            output = model(spectrum)
            loss = loss_fn(output, label)
            accuracy = (output.argmax(1) == label).sum()
            total_train_accuracy += accuracy.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print(f'Epoch:{i+1}  Train Accuracy:{total_train_accuracy/len(train_fold):.4f} Loss:{total_train_loss/len(train_fold):.4f} Lr:{lr}')

        # Validation
        total_valid_loss = 0
        total_accuracy = 0
        model.eval()
        valid_name_list = []
        valid_true_label = []
        valid_pred_label = []
        with torch.no_grad():
            for name, spectrum, label in valid_dataloader:
                valid_name_list.extend(name)
                valid_true_label.extend(label.numpy())
                spectrum = spectrum.to(device)
                label = label.to(device)
                output = model(spectrum)
                loss = loss_fn(output, label)
                total_valid_loss += loss.item()
                valid_pred_label.extend(output.argmax(1).cpu().detach().numpy())
                accuracy = (output.argmax(1) == label).sum()
                total_accuracy += accuracy.item()

        valid_accuracy = total_accuracy/len(val_fold)
        print(f'         Test Accuracy: {valid_accuracy:.4f} Loss: {total_valid_loss/len(val_fold):.4f}')
        
        if valid_accuracy > best_valid_accuracy:
            best_valid_accuracy = valid_accuracy
            model_best = model

        writer.add_scalar(f'Fold: {fold+1}/Loss', total_valid_loss, i+1)
        writer.add_scalar(f'Fold: {fold+1}/Accuracy', total_accuracy / len(val_fold), i+1)
        
    model_acc.append(best_valid_accuracy)
    model_list.append(model_best)

for i in range(num_folds):
    print(f'Fold {i+1} - Best Train Accuracy: {total_train_accuracy/len(train_fold):.4f}')
    print(f"Fold {i+1} - Best Valid Accuracy: {model_acc[i]:.4f}")

writer.close()

torch.save(model_list[np.argmax(model_acc)].state_dict(), 'inceptiontime.pt')

model = model_list[np.argmax(model_acc)]

test_data_t = Mydata(test_data)
test_dataloader_t = DataLoader(test_data_t, batch_size=1)

total_test_loss = 0
total_accuracy = 0
model.eval()
i = 0
judge=[]

with torch.no_grad():
    for name, spectrum, label in test_dataloader_t:
        i += 1
        spectrum = spectrum.to(device)
        label = label.to(device)
        output = model(spectrum)

        loss = loss_fn(output, label)
        total_test_loss += loss.item()
        accuracy = (output.argmax(1) == label).sum()
        total_accuracy = accuracy
        judge.append(total_accuracy.item())

print(len(test_data_t))
print(judge)

workbook = xlsxwriter.Workbook('inception_right.xlsx') 
worksheet = workbook.add_worksheet() 
rowTitle = judge
worksheet.write_row('A1', rowTitle) 
for i in range(0,len(rowTitle)):
    worksheet.write(0,i,rowTitle[i])
workbook.close()

# Test
def valid(model, testing_loader):
    
    steps = 0
    total_test_loss = 0
    total_test_accuracy = 0
    eval_preds, eval_labels, eval_probability = [], [], []
    softmax = nn.Softmax(dim=-1)

    model.eval()
    with torch.no_grad():
        for name, spectrum, label in testing_loader:
            
            spectrum = spectrum.to(device)
            label = label.to(device)
            output = model(spectrum)

            loss = loss_fn(output, label)
            total_test_loss += loss.item()

            steps += 1

            pred = torch.argmax(output, axis=1)
            accuracy = (pred == label).sum()
            total_test_accuracy += accuracy.item()

            eval_labels.extend(label.cpu().numpy())
            eval_preds.extend(pred.cpu().numpy())
            eval_probability.extend(softmax(output).cpu().numpy())
            
        epoch_loss = total_test_loss/steps
        te_accuracy = total_test_accuracy/len(test_data)
        print(f'Testing loss epoch: {epoch_loss}')
        print(f'Testing accuracy epoch: {te_accuracy}')
    
    return eval_labels, eval_preds, eval_probability

label, preds, probaility = valid(model, test_dataloader_t)

probaility_ = np.array(probaility)

print('acc: ',accuracy_score(label,preds))
print('auc: ',roc_auc_score(label, probaility_[:, 1]))
print('aupr: ',average_precision_score(label, probaility_[:, 1]))
print(confusion_matrix(label,preds))

#%%
##############################################################################
### Train and evaluate the InceptionTime model

# Data loading
train_data, valid_data = train_test_split(train_data, test_size=0.20, random_state=2)

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

train_dataset = Mydata(train_data)
valid_dataset = Mydata(valid_data)
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=64, shuffle=True)

model = Cmp_CNN()   
model.to(device)

loss_fn = nn.CrossEntropyLoss()
loss_fn.to(device)

from torch.optim.lr_scheduler import StepLR

learn_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate)
scheduler = StepLR(optimizer, step_size=50, gamma=0.1)

total_train_step = 0
total_valid_step = 0
epoch = 120

final_accuracy = 0

for i in range(epoch):
    print("-----------Epoch{}-----------".format(i + 1))
    total_train_loss = 0
    total_train_accuracy = 0

    model.train()
    for name, spectrum, label in train_dataloader:
        spectrum = spectrum.unsqueeze(1)
        spectrum = spectrum.to(device)
        label = label.to(device)
        output = model(spectrum)
        
        loss = loss_fn(output, label)
        accuracy = (output.argmax(1) == label).sum()
        total_train_accuracy += accuracy.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_train_step += 1

        total_train_loss += loss.item()
    scheduler.step()

    print("Epoch: {}, Train Accuracy: {}, Loss: {}".format(total_train_step, total_train_accuracy/len(train_data), total_train_loss/len(train_data)))
    print(optimizer.param_groups[0]['lr'])
    # print("-----------第{}轮测试-----------".format(i + 1))
        # Validation
    total_valid_loss = 0
    total_accuracy = 0
    model.eval()
    valid_name_list = []
    valid_true_label = []
    valid_pred_label = []
    with torch.no_grad():
        for name, spectrum, label in valid_dataloader:
            valid_name_list.extend(name)
            valid_true_label.extend(label.numpy())
            spectrum = spectrum.unsqueeze(1)
            spectrum = spectrum.to(device)
            label = label.to(device)
            output = model(spectrum)
            # print('output:', output, 'loss:', label)
            loss = loss_fn(output, label)
            total_valid_loss += loss.item()
            valid_pred_label.extend(output.argmax(1).cpu().detach().numpy())
            accuracy = (output.argmax(1) == label).sum()
            total_accuracy += accuracy

        print("Epoch: {}, Test Accuracy: {}, Loss: {}".format(i+1, total_accuracy.item()/len(valid_data), total_valid_loss/len(valid_data)))

    print(len(train_data), len(valid_data))

test_data_t = Mydata(test_data)
test_dataloader_t = DataLoader(test_data_t, batch_size=1)

# Test
total_test_loss = 0
total_accuracy = 0
model.eval()
i = 0
judge=[]

with torch.no_grad():
    for name, spectrum, label in test_dataloader_t:
        i += 1
        spectrum = spectrum.unsqueeze(1)
        spectrum = spectrum.to(device)
        label = label.to(device)
        output = model(spectrum)

        loss = loss_fn(output, label)
        total_test_loss += loss.item()
        accuracy = (output.argmax(1) == label).sum()
        total_accuracy = accuracy
        judge.append(total_accuracy.item())

workbook = xlsxwriter.Workbook('inception_right.xlsx') 
worksheet = workbook.add_worksheet() 
rowTitle = judge
worksheet.write_row('A1', rowTitle) 
for i in range(0,len(rowTitle)):
    worksheet.write(0,i,rowTitle[i])
workbook.close()

def valid(model, testing_loader):
    
    steps = 0
    total_test_loss = 0
    total_test_accuracy = 0
    eval_preds, eval_labels, eval_probability = [], [], []
    softmax = nn.Softmax(dim=-1)

    model.eval()
    with torch.no_grad():
        for name, spectrum, label in testing_loader:
            
            spectrum = spectrum.unsqueeze(1)
            spectrum = spectrum.to(device)
            label = label.to(device)
            output = model(spectrum)

            loss = loss_fn(output, label)
            total_test_loss += loss.item()

            steps += 1

            pred = torch.argmax(output, axis=1)
            accuracy = (pred == label).sum()
            total_test_accuracy += accuracy.item()

            eval_labels.extend(label.cpu().numpy())
            eval_preds.extend(pred.cpu().numpy())
            eval_probability.extend(softmax(output).cpu().numpy())

            
        epoch_loss = total_test_loss/steps
        te_accuracy = total_test_accuracy/len(test_data)
        print(f'Testing loss epoch: {epoch_loss}')
        print(f'Testing accuracy epoch: {te_accuracy}')
    
    return eval_labels, eval_preds, eval_probability

label, preds, probaility = valid(model, test_dataloader_t)

probaility_ = np.array(probaility)

accuracy = accuracy_score(label, preds)
auc = roc_auc_score(label, probaility_[:, 1])
aupr = average_precision_score(label, probaility_[:, 1])

print(f'acc: {accuracy:.4f}')
print(f'auc: {auc:.4f}')
print(f'aupr: {aupr:.4f}')
from sklearn.metrics import confusion_matrix
print(confusion_matrix(label,preds))