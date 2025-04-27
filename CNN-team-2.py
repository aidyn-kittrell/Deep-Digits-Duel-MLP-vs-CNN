import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ===================== Utility Functions ===================== #

def relu(x):
    """ ReLU activation function """
    return np.maximum(0, x)

def softmax(x):
    """ Softmax activation function """
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True)) # Subtract max for numerical stability
    return e_x / np.sum(e_x, axis=1, keepdims=True)


# ===================== Data Loading ===================== #
def dataloader(train_dataset, test_dataset, batch_size=128):
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
        self.input_size = input_size
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.fc_output_size = fc_output_size
        self.lr = lr

        # Calculate and store the convolutional output size
        self.conv_output_size = input_size - kernel_size + 1 # No padding

        # Initialize convolutional kernel weights and biases
        self.conv_kernels = np.random.randn(num_filters, kernel_size, kernel_size) * 0.01
        self.conv_biases = np.zeros(num_filters)

        # Initialize fully connected layer weights and biases
        self.fc_weights = np.random.randn(num_filters * self.conv_output_size * self.conv_output_size, fc_output_size) * 0.01
        self.fc_biases = np.zeros(fc_output_size)

    def convolve2d(self, x, kernel):
        """Perform a 2D convolution operation."""
        h, w = x.shape
        kh, kw = kernel.shape
        output_h = h - kh + 1
        output_w = w - kw + 1
        output = np.zeros((output_h, output_w))

        for i in range(output_h):
            for j in range(output_w):
                region = x[i:i+kh, j:j+kw]
                output[i, j] = np.sum(region * kernel)
        return output

    def forward(self, x):
        """ Forward propagation """
        batch_size = x.shape[0]
        conv_output_size = self.input_size - self.kernel_size + 1

        # Convolutional layer
        self.conv_outputs = np.zeros((batch_size, self.num_filters, conv_output_size, conv_output_size)) # store as an attribute
        for b in range(batch_size):
            for f in range(self.num_filters):
                # Extract the 2D image from the 3D tensor (assuming single channel)
                self.conv_outputs[b, f] = self.convolve2d(x[b, 0], self.conv_kernels[f]) + self.conv_biases[f]

        # Apply ReLU activation
        self.conv_outputs = relu(self.conv_outputs)

        # Flatten the output for the fully connected layer
        self.flattened = self.conv_outputs.reshape(batch_size, -1) # store as an attribute

        # Fully connected layer
        fc_outputs = np.dot(self.flattened, self.fc_weights) + self.fc_biases

        # Apply Softmax activation
        outputs = softmax(fc_outputs)

        return outputs

    def backward(self, x, y, pred):
        """ Backward propagation """
        batch_size = x.shape[0]

        # 1. one-hot encode the labels
        y_one_hot = np.zeros((batch_size, self.fc_output_size))
        y_one_hot[np.arange(batch_size), y] = 1

        # 2. Calculate softmax cross-entropy loss gradient
        dL_dfc_outputs = (pred - y_one_hot) / batch_size # gradient of loss w.r.t. softmax output
        
        # 3. Calculate fully connected layer gradient
        dL_dfc_weights = np.dot(self.flattened.T, dL_dfc_outputs) # gradient of loss w.r.t. fc weights
        dL_dfc_biases = np.sum(dL_dfc_outputs, axis=0) # gradient of loss w.r.t. fc biases
        dL_dflattened = np.dot(dL_dfc_outputs, self.fc_weights.T) # gradient of loss w.r.t. flattened input
        
        # 4. Backpropagate through ReLU
        dL_dconv_outputs = dL_dflattened.reshape(batch_size, self.num_filters, self.conv_output_size, self.conv_output_size)
        dL_dconv_outputs[self.conv_outputs <= 0] = 0 # gradient of loss w.r.t. conv outputs (ReLU)
        
        # 5. Calculate convolution kernel gradient
        dL_dconv_kernels = np.zeros_like(self.conv_kernels)
        dL_dconv_biases = np.sum(dL_dconv_outputs, axis=(0, 2, 3)) # gradient of loss w.r.t. conv biases

        for b in range(batch_size):
            for f in range(self.num_filters):
                for i in range(self.conv_output_size):
                    for j in range(self.conv_output_size):
                        # Extract the correct 2D region from the input
                        region = x[b, 0, i:i+self.kernel_size, j:j+self.kernel_size] # index the first channel
                        dL_dconv_kernels[f] += dL_dconv_outputs[b, f, i, j] * region
        
        # 6. Update parameters
        self.fc_weights -= self.lr * dL_dfc_weights
        self.fc_biases -= self.lr * dL_dfc_biases
        self.conv_kernels -= self.lr * dL_dconv_kernels
        self.conv_biases -= self.lr * dL_dconv_biases

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
    input_size = 28 # MNIST image size (28x28) (don't change)
    num_epochs = 5 # defined max number of epochs (don't change)
    fc_output_size = 10 # number of classes (0-9) (don't change)
    num_filters = 1 # number of filters in the convolutional layer (1 filter = 1 kernel) (don't change)
    
    kernel_size = 3 # kernel size (nxn)
    lr = 0.01 # learning rate

    # Third, initialize the model
    model = CNN(input_size=input_size, num_filters=num_filters, kernel_size=kernel_size, fc_output_size=fc_output_size, lr=lr)

    # Then, train the model
    for epoch in range(num_epochs):
        total_loss = 0

        for inputs, labels in train_loader: # define training phase for training model
            # Convert inputs and labels to numpy arrays
            x = inputs.numpy()
            y = labels.numpy()

            # Train the model on the current batch
            loss = model.train(x, y)
            total_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader)}") # print the loss for each epoch

    # Finally, evaluate the model
    correct_pred = 0
    total_pred = 0
    for inputs, labels in test_loader:
        # Convert inputs and labels to numpy arrays
        x = inputs.numpy()
        y = labels.numpy()

        # Perform forward pass to get predictions
        pred = model.forward(x) # the model refers to the model that was trained during the training phase
        predicted_labels = np.argmax(pred, axis=1)

        # Calculate accuracy
        correct_pred += np.sum(predicted_labels == y)
        total_pred += len(labels)
    print(f"Test Accuracy: {correct_pred/total_pred*100:.2f}%")

if __name__ == "__main__":
    main()
