import matplotlib.pyplot as plt
from math import floor
import pandas as pd
import torch
import numpy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # select first gpu
print("device = " + str(device))
torch.set_printoptions(sci_mode=False, edgeitems=5)

# data source: https://vincentarelbundock.github.io/Rdatasets/datasets.html

# todo:
    # find the right model; develop stopping procedure to stop training (when does overfitting start?) - maybe not
    # estimate model on full data? (when we have stopping procedure we need validation data)
    # save into csv: DV, treatment variable, model predictions
    # use R: run regression

mydata = pd.read_csv("data/Fatalities.csv")
exclude = ['fatal','nfatal','sfatal', 'fatal1517', 'nfatal1517', 'fatal1820', 'nfatal1820',
       'fatal2124', 'nfatal2124', 'afatal','spirits']
numeric = torch.tensor(mydata.drop(["state","breath","jail","service"]+exclude,axis=1).to_numpy(),dtype=torch.float).to(device)
binaryArray = [pd.factorize(mydata[i])[0].tolist() for i in ["breath","jail","service"]]
binaries = torch.transpose(torch.tensor(binaryArray,dtype=torch.float),0,1).to(device)
states = torch.tensor(pd.factorize(mydata["state"])[0])
categorical = torch.nn.functional.one_hot(states).to(device)
covariates = torch.cat((numeric,binaries,categorical),1)

dependent = torch.tensor(mydata["fatal"],dtype=torch.float).to(device)
regressor = torch.tensor(mydata["spirits"],dtype=torch.float).to(device) 

indices = torch.randperm(len(mydata))   # randomly reorder indices
nvalidation = 2000
nfolds = 10
foldsize = floor(covariates.size()[0]/nfolds)
folds = []
for k in range(nfolds+1):
    folds.append(k*foldsize)
folds[-1] = covariates.size()[0]

full_data = (dependent,covariates) # full data for ML, i.e. exclude private

n1 = 5 # size (output) of layer1
n2 = 5 # size (output) of layer2

class NonLinearNetwork(torch.nn.Module):
    def __init__(self):
        super(NonLinearNetwork,self).__init__()
        self.linear1 = torch.nn.Linear(covariates.size()[1],n1) # number of elements in layers are features
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(n1,n2)
        self.linear3 = torch.nn.Linear(n2,1)
    def forward(self,x): # neural network function (take input to create predictions)
        x = self.linear1(x)
        x = self.activation(x)
        #x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation(x)
        #x = self.dropout(x)
        x = self.linear3(x)
        return torch.squeeze(x)

class LinearNetwork(torch.nn.Module):
    def __init__(self):
        super(LinearNetwork,self).__init__()
        self.linear1 = torch.nn.Linear(covariates.size()[1],1)
    def forward(self,x):
        x = self.linear1(x)
        return torch.squeeze(x)

modelnonlinear = NonLinearNetwork().to(device)
modellinear = LinearNetwork().to(device)

loss_function = torch.nn.MSELoss()
optimizernonlinear = torch.optim.Adam(modelnonlinear.parameters(), lr=0.001)
optimizerlinear = torch.optim.Adam(modellinear.parameters(), lr=0.001)

def training_step(data,model,optimizer):
    model.train(True)                   # go into training mode
    optimizer.zero_grad()               # clear out old gradients
    y,x = data                          # y - targets, x - labels
    outputs = model(x)
    loss = loss_function(outputs,y)
    loss.backward()                    # backpropagation
    optimizer.step()
    return loss.item()

def validation_step(data,model,optimizer):
    model.eval()                        # go into validation mode, e.g. disable gradients and dropout layers
    with torch.no_grad():               # https://stackoverflow.com/questions/26342769/meaning-of-with-statement-without-as-keyword
        y,x = data
        outputs = model(x)
        loss = loss_function(outputs,y)
        return loss.item()

nsteps = 500
steps = [i for i in range(nsteps)]

def validate():                         # training loop: use training data, and validate with validation data   
    training_losses = [[0 for i in range(nsteps)] for i in range(2)]
    validation_losses = [[0 for i in range(nsteps)]  for i in range(2)]  
    models = [modelnonlinear, modellinear]
    optimizers = [optimizernonlinear, optimizerlinear]
    for k in range(nfolds-1):
        for model in models:
            for layer in model.children():
                if hasattr(layer, "reset_parameters"):
                    layer.reset_parameters()
        start_index = folds[k]
        end_index = folds[k+1]
        training_indices = torch.cat((indices[:start_index],indices[end_index:]),0)
        validation_indices = indices[start_index:end_index]
        training_data = (dependent[training_indices], covariates[training_indices, :])
        validation_data = (dependent[validation_indices], covariates[validation_indices, :])
        for step in steps:
            for i in range(2):
                model = models[i]
                optimizer = optimizers[i]
                training_loss = training_step(training_data,model,optimizer)
                validation_loss = validation_step(validation_data,model,optimizer)
                training_losses[i][step] += training_loss
                validation_losses[i][step] += validation_loss
                if step % 100 == 0 and i==0:
                    print("fold:", k+1,"   step:", step,"   training loss:",round(training_loss,4),"   validation loss:",round(validation_loss,4))
    training_losses_nonlinear = [loss/nfolds for loss in training_losses[0]]
    training_losses_linear = [loss / nfolds for loss in training_losses[1]]
    validation_losses_nonlinear = [loss / nfolds for loss in validation_losses[0]]
    validation_losses_linear = [loss / nfolds for loss in validation_losses[1]]
    return training_losses_nonlinear,training_losses_linear,validation_losses_nonlinear,validation_losses_linear

training_losses_nonlinear,training_losses_linear,validation_losses_nonlinear,validation_losses_linear = validate() 

def plot(training_losses_nonlinear,training_losses_linear,validation_losses_nonlinear,validation_losses_linear):
    xlim = 75
    plt.ion()                           # ion - interactive mode on - non-blocking graphing
    plt.cla()
    #plt.plot(steps[xlim:],training_losses_nonlinear[xlim:],label="training losses nonlinear",linewidth=4)
    #plt.plot(steps[xlim:],training_losses_linear[xlim:],label="training losses linear",linewidth=4)
    plt.plot(steps[xlim:],validation_losses_nonlinear[xlim:], label="validation losses nonlinear", linewidth=2)
    plt.plot(steps[xlim:],validation_losses_linear[xlim:], label="validation losses linear", linewidth=2)
    #totallosses = training_losses_nonlinear+validation_losses_nonlinear+training_losses_linear+validation_losses_linear
    #plt.ylim(min(totallosses),0.55)
    plt.legend()
    plt.show()
    plt.pause(0.01)

plot(training_losses_nonlinear,training_losses_linear,validation_losses_nonlinear,validation_losses_linear)

def train(): 
    models = [modelnonlinear, modellinear]
    optimizers = [optimizernonlinear, optimizerlinear]
    for model in models:
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    for step in steps:
        for i in range(2):
            model = models[i]
            optimizer = optimizers[i]
            loss = training_step(full_data,model,optimizer)
            if step % 100 == 0 and i==0:
                print("step:", step,"   loss:",round(loss,4))

train()

# save results
modelnonlinear.eval()
modellinear.eval()
results = {
    "dependent": dependent.cpu().detach().tolist(),
    "regressor": regressor.cpu().detach().tolist(),
    "predictionNonlinear": modelnonlinear(covariates).cpu().detach().tolist(),
    "predictionLinear": modellinear(covariates).cpu().detach().tolist()
}
pd.DataFrame(data=results).to_csv("./predictions.csv",index=False)
print("CSV Written")
