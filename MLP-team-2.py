import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def sigmoid(x):  # manually define the sigmoid
    return 1 / (1 + np.exp(-x))

def softmax(x):  # define the softmax 
    x_shift = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x_shift)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def cross_entropy_loss(probs, labels):
    batch_size = labels.shape[0]
    clipped = np.clip(probs, 1e-12, 1.0)
    log_likelihood = -np.log(clipped[np.arange(batch_size), labels])
    return np.mean(log_likelihood)

def dataloader(train_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("The number of training data:", len(train_dataset))
    print("The number of testing data:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

class MLP:
    def __init__(self, input_size, hidden_size, output_size,lr):  # 
        self.lr=lr
        self.W1= np.random.randn(input_size, hidden_size)*0.01
        self.B1= np.zeros((1, hidden_size))
        self.W2= np.random.randn(hidden_size, output_size) * 0.01
        self.B2= np.zeros((1, output_size))
    
    def forward(self, x):  # forward propagation to get predictions
        self.Z1 = x.dot(self.W1) + self.B1
        self.A1 = sigmoid(self.Z1)

        self.Z2 = self.A1.dot(self.W2) + self.B2
        outputs = softmax(self.Z2)
        return outputs
    
    def backward(self, x, y, pred):
        # one-hot encode the labels
        batch_size = x.shape[0]
        y_onehot = np.zeros_like(pred)
        y_onehot[np.arange(batch_size), y] = 1

        # compute the gradients
        dZ2 = (pred - y_onehot) / batch_size
        dW2 = self.A1.T.dot(dZ2)
        dB2 = np.sum(dZ2, axis=0)

        dA1 = dZ2.dot(self.W2.T)
        dZ1 = dA1 * (self.A1 * (1 - self.A1))

        dW1 = x.T.dot(dZ1)
        dB1 = np.sum(dZ1, axis=0)

        # update the weights and biases
        self.W2 -= self.lr * dW2
        self.B2 -= self.lr * dB2
        self.W1 -= self.lr * dW1
        self.B1 -= self.lr * dB1

    def train(self, x,y):
        # call forward function
        pred = self.forward(x)
        # calculate loss
        loss = cross_entropy_loss(pred, y)
        # call backward function
        self.backward(x, y, pred)
        return loss

def main():
    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28*28  # MNIST images are 28x28 pixels
    output_size = 10

    num_epochs = 100
    hidden_size = 512
    lr = 0.09
    
    model = MLP(input_size, hidden_size, output_size, lr)
    
    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader:  # define training phase for training model
            x = inputs.view(-1, input_size).numpy()
            y = labels.numpy()
            loss = model.train(x, y)
            total_loss += loss
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}") # print the loss for each epoch

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.view(-1, input_size).numpy()
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the raining phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred}")

if __name__ == "__main__":  # Program entry
    main()
