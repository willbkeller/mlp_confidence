import argparse
from random import shuffle
import torch
from torch import nn 
import torch.nn.functional as F
from torchvision import datasets, transforms
from sklearn.metrics import confusion_matrix

if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device = torch.device('cpu')

print('Pytorch Version: ', torch.__version__, ' Device: ', device)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = False
        self.fc1=nn.Linear(784, 1000)
        self.fc2=nn.Linear(1000, 256)
        self.fc3=nn.Linear(256, 64)
        self.fc4=nn.Linear(64, 128)
        self.fc5=nn.Linear(128,10)
        self.fc_drop = nn.Dropout(0.3)
        #self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        op = x.view(-1, self.fc1.in_features)
        op = F.relu(self.fc1(op))
        op = F.relu(self.fc2(op))
        op = F.relu(self.fc3(op))
        op = F.relu(self.fc4(op))

        if self.dropout:
            op = self.fc_drop(op)

        pred = self.fc5(op)
        #pred = self.softmax(op)
        return pred

mlp_1 = MLP()

optimizer = torch.optim.SGD(mlp_1.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))
    ])), batch_size=128, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data/mnist', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,))
    ])), batch_size=128, shuffle=True
)

def train(model, epoch, best_acc):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.view(-1, 784)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '+
                  f'({(100.*batch_idx/len(train_loader)):.2f}%)]\tLoss: {loss.item():.6f}')
        
    model.eval()
    test_loss = 0
    correct=0

    with torch.no_grad():
        for data, target in test_loader:
            data=data.view(-1,784)
            output=model(data)
            test_loss+=criterion(output,target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct+= pred.eq(target.data.view_as(pred)).sum()

            output = (torch.max(torch.exp(output),1)[1]).data.cpu().numpy()
            target = target.data.cpu().numpy()
    
    test_loss/=len(test_loader.dataset)
    print(f'Test Set: Avg. Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({(100.*correct/len(test_loader.dataset)):.2f}%) ')
    if correct/len(test_loader.dataset) > best_acc: 
        best_acc = correct/len(test_loader.dataset)
        print('better')
        torch.save(model.state_dict(), f"../saved_models/mlp_3_resume.pt")
    return best_acc
        

                  
def test(model):
    y_pred=[]
    y_true=[]
    model.eval()
    test_loss = 0
    correct=0

    with torch.no_grad():
        for data, target in test_loader:
            data=data.view(-1,784)
            output=model(data)
            test_loss+=criterion(output,target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct+= pred.eq(target.data.view_as(pred)).sum()

            output = (torch.max(torch.exp(output),1)[1]).data.cpu().numpy()
            y_pred.extend(output)
            target = target.data.cpu().numpy()
            y_true.extend(target)
    
    test_loss/=len(test_loader.dataset)
    print(f'Test Set: Avg. Loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({(100.*correct/len(test_loader.dataset)):.2f}%) ')
    conf_mat = confusion_matrix(y_pred, y_true)
    print(conf_mat)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MLP_0")
    parser.add_argument('--epochs', type=int, default=25, metavar='N', help='number of epochs for training (default: 25)')

    parser.add_argument('--train', action='store_true', default=False, help='train model')
    parser.add_argument('--test', action='store_true', default=False, help='test model')
    args=parser.parse_args()
    
    best_acc=0.0

    if args.train:
        for epoch in range(1, args.epochs+1):
            print('Best:',best_acc)
            best_acc = train(mlp_1, epoch, best_acc)
    
    if args.test:
        mlp_1.load_state_dict(torch.load('../saved_models/mlp_3_resume.pt'))
        test(mlp_1)