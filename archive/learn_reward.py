import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
import assign_preferences
# simple neural network with one linear layer
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc = nn.Linear(2, 1, bias=False)  # no bias since we're just interested in the weights
    
    def forward(self, x):
        return self.fc(x)

def custom_loss(output0, output1, preference):
    output0 = output0.squeeze()
    output1 = output1.squeeze()
    
    # calculate predicted preferences based on rewards
    predicted_preferences = torch.zeros_like(preference)
    predicted_preferences[output0 < output1] = 1
    predicted_preferences[output0 > output1] = 0
    predicted_preferences[output0 == output1] = -1  # or could use some other value like 0.5?
    
    valid_indices = (preference != -1)
    filtered_preferences = preference[valid_indices]
    filtered_predicted_preferences = predicted_preferences[valid_indices]
    filtered_preferences = filtered_preferences.long()
    
    # stack the predicted preferences as logits for cross-entropy loss
    logits = torch.stack([output0, output1], dim=1)[valid_indices]
    
    loss = F.cross_entropy(logits, filtered_preferences)
    
    return loss

# def custom_loss(output0, output1, preference):
#     # loss that is just linear function of distance from correct preference   
#     output0 = output0.squeeze()
#     output1 = output1.squeeze()

#     loss = torch.tensor(0.0, dtype=torch.float32)

#     for o0, o1, p in zip(output0, output1, preference):
#         if p == 1:  # Preference for second output
#             loss += torch.relu(o0 - o1)
#         elif p == 0:  # Preference for first output
#             loss += torch.relu(o1 - o0)
#         elif p == -1:  # Equally preferred
#             loss += torch.abs(o0 - o1)

#     return loss

def main():
    # X0 and X1 are tensors of shape (num_samples, num_features), y is a tensor of shape (num_samples,)
    data_file = sys.argv[1]
    pref_file = sys.argv[2]

    with open(data_file, 'rb') as f:
        segments, feature_vectors = pickle.load(f)
    with open(pref_file, 'rb') as f:
        preferences = pickle.load(f)

    # this is just bc our first two features in generated dataset are redundant
    feature_vectors = [(
        pair1[1:],  
        pair2[1:]  
    ) for pair1, pair2 in feature_vectors]

    X0 = torch.tensor( np.array(list(pair[0] for pair in feature_vectors)), dtype=torch.float32)  
    X1 = torch.tensor( np.array(list(pair[1] for pair in feature_vectors)), dtype=torch.float32)  
    y = torch.tensor(preferences, dtype=torch.float32)   
    model = SimpleNN()
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # training loop
    num_epochs = 2000
    loss_values = []
    for epoch in range(num_epochs):
        model.train()
        
        # forward pass
        output0 = model(X0)
        output1 = model(X1)
        
        # compute loss
        loss = custom_loss(output0, output1, y)
        loss_values.append(loss.item())
        if loss == 0:
            break

        # backward pass 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
            weights = model.fc.weight.data
            print(f'Learned weights: {weights}')

    weights = model.fc.weight.data
    print(f'Learned weights: {weights}')

    # plot loss
    plt.plot(loss_values)
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.title('Cross-Entropy Loss over Epochs')
    plt.savefig('loss_plot.png')

    # evaluate preferences
    args = ['assign_preferences.py', 'test_segment_dataset', 'segment_preferences', weights]
    sys.argv = args
    assign_preferences.main()

if __name__ ==  "__main__":
    main()
