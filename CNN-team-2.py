import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ===================== Utility Functions ===================== #
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / e_x.sum(axis=1, keepdims=True)

# ===================== Data Loading ===================== #
def dataloader(train_dataset, test_dataset, batch_size=64):
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader

def load_data():
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    train_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root="./data/mnist", train=False, download=True, transform=transform)
    print("Training samples:", len(train_dataset))
    print("Testing samples:", len(test_dataset))
    return dataloader(train_dataset, test_dataset)

# ===================== CNN Structure ===================== #
class CNN:
    def __init__(self, input_size, num_filters, kernel_size, fc_output_size, lr):
        self.kernel_size = kernel_size
        self.lr = lr

        # Intialize convolution kernel (3x3)
        self.conv_kernel = np.random.randn(kernel_size, kernel_size) * 0.1
        self.conv_bias = np.zeros(1)

        # Fully connected layer parameters
        fc_input_size = (input_size - kernel_size + 1) ** 2 * num_filters
        self.W = np.random.randn(fc_input_size, fc_output_size) * 0.1
        self.b = np.zeros(fc_output_size)

    def forward(self, x):
        """ Forward propagation """
        batch_size = x.shape[0]
        output_size = 28 - self.kernel_size + 1
        conv_output = np.zeros((batch_size, output_size, output_size))

        # Convolution operation
        for i in range(output_size):
            for j in range(output_size):
                conv_output[:, i, j] = np.sum(x[:, i:i+self.kernel_size, j:j+self.kernel_size] * self.conv_kernel, axis=(1,2)) + self.conv_bias

        # ReLU activation
        conv_output = relu(conv_output)

        # Flatten for fully connected layer
        self.flattened = conv_output.reshape(batch_size, -1)

        # Fully connected layer
        logits = np.dot(self.flattened, self.W) + self.b

        # Softmax activation
        outputs = softmax(logits)

        return outputs

    def backward(self, x, y, pred):
        """ Backward propagation """
        batch_size = x.shape[0]
        output_size = 28 - self.kernel_size + 1

        # 1. one-hot encode the labels
        one_hot_y = np.eye(10)[y]
        
        # 2. Calculate softmax cross-entropy loss gradient
        grad_loss = (pred - one_hot_y) / batch_size
        
        # 3. Calculate fully connected layer gradient
        grad_W = np.dot(self.flattened.T, grad_loss)
        grad_b = np.sum(grad_loss, axis=0)
        
        # 4. Backpropagate through ReLU
        grad_conv = np.dot(grad_loss, self.W.T) * (self.flattened > 0)
        grad_conv = grad_conv.reshape(batch_size, output_size, output_size)
        
        # 5. Calculate convolution kernel gradient
        grad_kernel = np.zeros_like(self.conv_kernel)
        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                grad_kernel[i, j] = np.sum(x[:, i:i+output_size, j:j+output_size] * grad_conv) / batch_size
        
        # 6. Update parameters
        self.W -= self.lr * grad_W
        self.b -= self.lr * grad_b
        self.conv_kernel -= self.lr * grad_kernel

    def train(self, x, y):
        # call forward function
        pred = self.forward(x)
        
        # calculate loss
        loss = -np.sum(np.log(pred[np.arange(len(y)), y])) / len(y)
        
        # call backward function
        self.backward(x, y, pred)

        return loss

# ===================== Training Process ===================== #
def main():
    # First, load data
    train_loader, test_loader = load_data()

    # Second, define hyperparameters
    input_size = 28
    num_filters = 1
    num_epochs = 5
    kernel_size = 3
    lr = 0.01
    
    model = CNN(input_size, num_filters, kernel_size=kernel_size, fc_output_size=10, lr=lr)

    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader:  # define training phase for training model
            x = inputs.numpy().squeeze(1) # remove the channel dimension
            y = labels.numpy()
            loss = model.train(x, y)
            total_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}") # print the loss for each epoch

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        x = inputs.numpy().squeeze(1)  # keep original image dimensions
        y = labels.numpy()
        pred = model.forward(x)  # the model refers to the model that was trained during the raining phase
        predicted_labels = np.argmax(pred, 1)
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred*100:.2f}%")

if __name__ == "__main__":
    main()
