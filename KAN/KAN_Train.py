import torch
from kan import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = np.loadtxt("data.csv", delimiter=",",skiprows=1,usecols=(0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18))
# create a KAN: 2D inputs, 1D output, and 5 hidden neurons. cubic spline (k=3), 5 grid intervals (grid=5).

dataset = {}
dataset['train_input'] = torch.Tensor(data[:64,:15]).to(device)
dataset['test_input'] = torch.Tensor(data[64:,:15]).to(device)
dataset['train_label'] = torch.Tensor(data[:64,16].reshape([-1,1])).to(device)
dataset['test_label'] = torch.Tensor(data[64:,16].reshape([-1,1])).to(device)

model = KAN(width=[15,7,1], grid=3, k=3, seed=1, device=device)

# train the model
model.fit(dataset, opt="Adam", steps=10, lamb=0.0005)
model = model.prune()
model.plot()
plt.show()
model.fit(dataset, opt="LBFGS", steps=10)
model = model.prune()
model.plot()
plt.show()