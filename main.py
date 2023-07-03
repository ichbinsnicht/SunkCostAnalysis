import pandas as pd
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # select first gpu
print("device = " + str(device))
torch.set_printoptions(sci_mode=False, edgeitems=5)

mydata = pd.read_csv("data/DoctorVisits-Column1.csv")


visits = torch.tensor(mydata["visits"],dtype=torch.float).to(device)
gender = torch.tensor(pd.factorize(mydata["gender"])[0],dtype=torch.float).to(device)
age = torch.tensor(mydata["age"],dtype=torch.float).to(device)
income = torch.tensor(mydata["income"],dtype=torch.float).to(device)
illness = torch.tensor(mydata["illness"],dtype=torch.float).to(device)
reduced = torch.tensor(mydata["reduced"],dtype=torch.float).to(device)
health = torch.tensor(mydata["health"],dtype=torch.float).to(device)
private = torch.tensor(pd.factorize(mydata["private"])[0],dtype=torch.float).to(device)
freepoor = torch.tensor(pd.factorize(mydata["freepoor"])[0],dtype=torch.float).to(device)
freerepat = torch.tensor(pd.factorize(mydata["freerepat"])[0],dtype=torch.float).to(device)
nchronic = torch.tensor(pd.factorize(mydata["nchronic"])[0],dtype=torch.float).to(device)
lchronic = torch.tensor(pd.factorize(mydata["lchronic"])[0],dtype=torch.float).to(device)
covariates = torch.stack((gender,age,income,illness,reduced,health,freepoor,freerepat,nchronic,lchronic),1)
# DV - target; regressors - labels

# more parameters than variables: how do we check for overfitting? ==> separate data into training and validation data

indices = torch.randperm(len(mydata))   # randomly reorder indices
nvalidation = 2000
validation_indices = indices[:2000]
training_indices = indices[2000:]
validation_data = (visits[validation_indices],covariates[validation_indices,:])
training_data = (visits[training_indices],covariates[training_indices,:])
full_data = (visits,covariates)         # full data for ML, i.e. exclude private

n1 = 15             # size (output) of layer1
n2 = 10             # size (output) of layer2

# Neural networks with classes, # MyNetwork: class, self: instance of the class
class MyNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNetwork,self).__init__()
        self.linear1 = torch.nn.Linear(covariates.size()[1],n1) # number of elements in layers are features
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(n1,n2)
        self.linear3 = torch.nn.Linear(n2,1)
    def forward(self,x):                # neural network function (take input to create predictions)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x

model = MyNetwork()
#for p in model.parameters():
#    print(p)
# print(model.linear1.weight)
# print(model.linear1.bias)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

