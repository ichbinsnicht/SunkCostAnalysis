import matplotlib.pyplot as plt
from math import floor
import pandas as pd
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # select first gpu
print("device = " + str(device))
torch.set_printoptions(sci_mode=False, edgeitems=5)

# todo:
    # find the right model; develop stopping procedure to stop training (when does overfitting start?) - maybe not
    # estimate model on full data? (when we have stopping procedure we need validation data)
    # save into csv: DV, treatment variable, model predictions
    # use R to run regression

# data source: https://vincentarelbundock.github.io/Rdatasets/datasets.html
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
# k-fold validation: k - leave one out cross-validation, e.g. 5190 sets, covariates.size()

indices = torch.randperm(len(mydata))   # randomly reorder indices
nvalidation = 2000
nfolds = 10
foldsize = floor(covariates.size()[0]/nfolds)
folds = []
for k in range(nfolds+1):
    folds.append(k*foldsize)
folds[-1] = covariates.size()[0]

full_data = (visits,covariates)         # full data for ML, i.e. exclude private

n1 = 4             # size (output) of layer1
n2 = 3             # size (output) of layer2

# Neural networks with classes, # MyNetwork: class, self: instance of the class
class MyNetwork(torch.nn.Module):
    def __init__(self):
        super(MyNetwork,self).__init__()
        self.linear1 = torch.nn.Linear(covariates.size()[1],n1) # number of elements in layers are features
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(n1,n2)
        self.linear3 = torch.nn.Linear(n2,1)
    def forward(self,x):                # neural network function (take input to create predictions)
        x = self.linear1(x)
        x = self.activation(x)
        #x = self.dropout(x)
        x = self.linear2(x)
        x = self.activation(x)
        #x = self.dropout(x)
        x = self.linear3(x)
        return torch.squeeze(x)

model = MyNetwork().to(device)
#for p in model.parameters():
#    print(p)
# print(model.linear1.weight)
# print(model.linear1.bias)

loss_function = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

######## Objects and Functions
# f.x - accessing a property of an object that may be a function
# f(x) - functions
# f.g(x) = applying a method to an object
# in python objects are also functions

# compare shape of targets and output via loss function (no overfitting but still fitting)

output = model(covariates)

# training step: compute output, compute gradients, and adjust parameters with training data

def training_step(data):
    model.train(True)                   # go into training mode
    optimizer.zero_grad()               # clear out old gradients
    y,x = data                          # y - targets, x - labels
    outputs = model(x)
    loss = loss_function(outputs,y)
    loss.backward()                    # backpropagation
    optimizer.step()
    return loss.item()

def validation_step(data):
    model.eval()                        # go into validation mode, e.g. disable gradients and dropout layers
    with torch.no_grad():               # https://stackoverflow.com/questions/26342769/meaning-of-with-statement-without-as-keyword
        y,x = data
        outputs = model(x)
        loss = loss_function(outputs,y)
        return loss.item()

def validate():                         # training loop: use training data, and validate with validation data
    nsteps = 1000
    training_losses = [0 for i in range(nsteps)]
    validation_losses = [0 for i in range(nsteps)]
    steps = [i for i in range(nsteps)]
    for k in range(nfolds-1):
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        start_index = folds[k]
        end_index = folds[k+1]
        training_indices = torch.cat((indices[:start_index],indices[end_index:]),0)
        validation_indices = indices[start_index:end_index]
        training_data = (visits[training_indices], covariates[training_indices, :])
        validation_data = (visits[validation_indices], covariates[validation_indices, :])
        for step in steps:
            training_loss = training_step(training_data)
            validation_loss = validation_step(validation_data)
            training_losses[step] += training_loss
            validation_losses[step] += validation_loss
            if step % 100 == 0:
                print("fold:", k+1,"   step:", step,"   training loss:",round(training_loss,4),"   validation loss:",round(validation_loss,4))
    training_losses = [loss/nfolds for loss in training_losses]
    validation_losses = [loss / nfolds for loss in validation_losses]
    plt.ion()                           # ion - interactive mode on - non-blocking graphing
    plt.cla()
    plt.plot(steps,training_losses,label="training losses",linewidth=4)
    plt.plot(steps, validation_losses, label="validation losses", linewidth=2)
    plt.ylim(min(training_losses+validation_losses),0.55)
    plt.legend()
    plt.show()
    plt.pause(0.01)

def validate_simple():
    nsteps = 500
    training_losses = []
    validation_losses = []
    steps = []
    training_indices = indices[2000:]
    validation_indices = indices[:2000]
    training_data = (visits[training_indices], covariates[training_indices, :])
    validation_data = (visits[validation_indices], covariates[validation_indices, :])
    for step in range(nsteps):
        training_loss = training_step(training_data)
        validation_loss = validation_step(validation_data)
        training_losses.append(training_loss)
        validation_losses.append(validation_loss)
        steps.append(step)
        if step % 100 == 0:
            print("step:", step,"   training loss:",round(training_loss,4),"   validation loss:",round(validation_loss,4))
    plt.ion()
    plt.cla()
    plt.plot(steps,training_losses,label="training losses",linewidth=4)
    plt.plot(steps, validation_losses, label="validation losses", linewidth=2)
    plt.legend()
    plt.show()
    plt.pause(0.01)

for layer in model.children():
    if hasattr(layer,"reset_parameters"):
        layer.reset_parameters()

validate()