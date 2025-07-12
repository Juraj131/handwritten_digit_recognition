import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from architecture import calculate_accuracy


def calculate_accuracy(loader, model):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            images = images.view(images.size(0), -1)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    accuracy = 100 * correct / total
    return accuracy, all_preds, all_labels

# Definícia architektúry modelu
class MLP_net(nn.Module):
    def __init__(self, input_size, hidden_1_size, hidden_2_size, hidden_3_size, output_size):
        super(MLP_net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_1_size)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_1_size, hidden_2_size)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_2_size, hidden_3_size)
        self.dropout3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(hidden_3_size, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

# Example usage
input_size = 28 * 28  # Assuming input images are 28x28
hidden_1_size = 512
hidden_2_size = 256
hidden_3_size = 128
output_size = 10  # Assuming 10 classes

device = "cpu"
mymodel = MLP_net(input_size, hidden_1_size, hidden_2_size, hidden_3_size, output_size).to(device)
mymodel.load_state_dict(torch.load('saved_weights\mymodel_3.pth', map_location=device, weights_only=True))

# Teraz môžete použiť model na predikcie alebo ďalšie spracovanie
mymodel.eval()

# Definovanie transformácií
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),  # Zmena veľkosti obrázkov na 28x28
    transforms.ToTensor(),        # Konverzia obrázkov na tenzory
    transforms.Normalize((0.5,), (0.5,)),  # Normalizácia obrázkov
])

# Načítanie datasetu
dataset = datasets.ImageFolder(root=r'C:\Users\juraj\OneDrive\Documents\UNI_BTB\5.semester\UIM\final_projekt2\number_recogniton\final_test_data', transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# Testovanie všetkých obrázkov v datasete
correct = 0
total = 0

with torch.no_grad():
    for images, labels in dataloader:
        images = images.view(-1, 28*28)  # Preveďte na správny tvar
        outputs = mymodel(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the model on the dataset: {100 * correct / total}%')

val_accuracy, val_preds, vals_labels = calculate_accuracy(dataloader, mymodel)
print(f'Accuracy by calculate_accuracy func: {val_accuracy}%')
# Vytvoření a vykreslení confusion matrix pro validační data
val_cm = confusion_matrix(vals_labels, val_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=val_cm, display_labels=dataset.classes)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix - Validation Data')
plt.show()