import kagglehub
import os
from PIL import Image
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
import seaborn as sns

DATASET_LINK = "apollo2506/eurosat-dataset"
dataset_path = kagglehub.dataset_download(DATASET_LINK)
dataset_jpg_path = os.path.join(dataset_path, "EuroSAT")

train_df = pd.read_csv(os.path.join(dataset_jpg_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(dataset_jpg_path, 'test.csv'))
validation_df = pd.read_csv(os.path.join(dataset_jpg_path, 'validation.csv'))


class EuroSATDataset(Dataset):
    # Initialize the Dataset class:
    # Инициализируем класс Dataset:
    def __init__(self, dataframe, root_dir, transform=None):
        self.dataframe = dataframe  # dataframe is a DataFrame with information about paths to images and labels | dataframe — это DataFrame с информацией о путях к изображениям и метках.
        self.root_dir = root_dir  # root_dir is the root directory where the images are located | root_dir — это корневая директория, где находятся изображения
        self.transform = transform  # transform — these are transformations that will be applied to images (resize) | transform — это преобразования, которые будут применяться к изображениям (resize)

    def __len__(self):
        return len(
            self.dataframe)  # We return the number of examples in the dataset (rows in the dataframe) | Возвращаем количество примеров в датасете (строк в dataframe).

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx, 1]  # image path (first column) | путь к изображению (первая колонка)
        label = self.dataframe.iloc[idx, 2]  # class label (second column) | метка класса (вторая колонка)

        img_path = os.path.join(self.root_dir,
                                img_name)  # Building the full path to the image | Строим полный путь к изображению

        image = Image.open(img_path)  # Opening the image | Открываем изображение

        # If transformations are set, we apply them to the image.
        # Если трансформации заданы, применяем их к изображению.
        if self.transform:
            image = self.transform(image)

        return image, label  # Returning the tuple (image, label) | Возвращаем кортеж (изображение, метка)


transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
])

train_dataset = EuroSATDataset(train_df, dataset_jpg_path, transform=transform)
test_dataset = EuroSATDataset(test_df, dataset_jpg_path, transform=transform)
val_dataset = EuroSATDataset(validation_df, dataset_jpg_path, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

class_names = train_df[['Label', 'ClassName']]\
              .drop_duplicates()\
              .sort_values('Label')\
              .ClassName.unique()

class EuroSATModel(nn.Module):
    def __init__(self):
        super(EuroSATModel, self).__init__()

        # The first convolutional layer (16 filters, 3x3 core, padding 1, maxpowling 2x2)
        # Первый сверточный слой (16 фильтров, ядро 3x3, padding 1, макспулинг 2x2)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional layer (32 filters, 3x3 core, padding 1, maxpowling 2x2)
        # Второй сверточный слой (32 фильтра, ядро 3x3, padding 1, макспулинг 2x2)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # The third convolutional layer (64 filters, 3x3 core, padding 1, maxpowling 2x2)
        # Третий сверточный слой (64 фильтра, ядро 3x3, padding 1, макспулинг 2x2)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers after convolutional (flattening)
        # Полносвязные слои после сверточных (flattening)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)  # The size of the 64 filter is 8x8 after all pooling (if the input image is 64x64) | Размер 64 фильтра, 8x8 после всех пулингов (если входное изображение 64x64)
        self.fc2 = nn.Linear(256, 10)  # 10 classes | 10 классов

    def forward(self, x):
        # We go through the first convolutional layer and pooling
        # Проходим через первый сверточный слой и пулинг
        x = self.pool1(torch.relu(self.conv1(x)))

        # We go through the second convolutional layer and pooling
        # Проходим через второй сверточный слой и пулинг
        x = self.pool2(torch.relu(self.conv2(x)))

        # We go through the third convolutional layer and pooling
        # Проходим через третий сверточный слой и пулинг
        x = self.pool3(torch.relu(self.conv3(x)))

        # Converting the tensor to the form (batch_size, 64 * 8 * 8 )
        # Переводим тензор в форму (batch_size, 64 * 8 * 8)
        x = x.view(-1, 64 * 8 * 8)

        # Fully connected layer
        # Полносвязный слой
        x = torch.relu(self.fc1(x))

        # Output layer (classification)
        # Выходной слой (классификация)
        x = self.fc2(x)

        return x


def evaluate(model, val_loader):
    model.eval()  # Setting the model to evaluation mode | Устанавливаем модель в режим оценки
    correct = 0  # A variable for calculating the number of correct predictions | Переменная для подсчета количества правильных прогнозов
    total = 0  # A variable for counting the total number of samples | Переменная для подсчета общего числа образцов
    val_loss = 0.0  # Variable for calculating validation losses| Переменная для подсчета потерь на валидации
    criterion = nn.CrossEntropyLoss()  # Loss function for multiclass classification | Функция потерь для многоклассовой классификации

    with torch.no_grad():  # Disabling the calculation of gradients to improve performance during evaluation | Отключаем вычисление градиентов для улучшения производительности во время оценки
        for inputs, labels in val_loader:  # Iterate over the validation data loader | Итерируем по валидационному загрузчику данных
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)  # Run the input data through the model to obtain forecasts | Прогоняем входные данные через модель для получения прогнозов

            loss = criterion(outputs, labels)  # Calculate losses (errors) for the current data portion | Вычисляем потери (ошибки) для текущей порции данных
            val_loss += loss.item()  # Add losses to the total amount | Добавляем потери к общей сумме

            # Determining the number of correct forecasts | Определяем количество правильных прогнозов
            _, predicted = torch.max(outputs, 1)  # Choosing the class with the highest probability | Выбираем класс с максимальной вероятностью
            total += labels.size(0)  # Increasing the total number of samples | Увеличиваем общее количество образцов
            correct += (predicted == labels).sum().item()  # Let's summarize the number of correct predictions | Суммируем количество правильных прогнозов

    accuracy = 100 * correct / total  # Calculate the accuracy as the percentage of correct answers | Вычисляем точность как процент правильных ответов
    avg_val_loss = val_loss / len(val_loader)  # Average losses in the validation sample | Средние потери на валидационной выборке

    return accuracy, avg_val_loss  # Return accuracy and average losses | Возвращаем точность и средние потери

# Using GPU | Использование видеокарты
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Creating a model
# Создаем модель
model = EuroSATModel().to(device)

# Defining the loss function (Cross-Entropy Loss)
# Определяем функцию потерь (Cross-Entropy Loss)
criterion = nn.CrossEntropyLoss()

# Defining an optimizer (for example, Adam)
# Определяем оптимизатор (например, Adam)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
train_losses = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    model.train()  # Setting the model to training mode | Устанавливаем модель в режим тренировки
    running_loss = 0.0

    # Model training
    # Тренировка модели
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        # Zeroing out the gradients of the previous steps
        # Обнуляем градиенты предыдущих шагов
        optimizer.zero_grad()

        # Running through the model
        # Прогоняем через модель
        outputs = model(inputs)

        # Calculating losses
        # Вычисляем потери
        loss = criterion(outputs, labels)

        # Error propagation back
        # Обратное распространение ошибки
        loss.backward()

        # Updating the model parameters
        # Обновляем параметры модели
        optimizer.step()

        # Summarize the losses
        # Суммируем потери
        running_loss += loss.item()

    # Assessment for validation
    # Оценка на валидации
    val_accuracy, val_loss = evaluate(model, val_loader)

    val_accuracy, val_loss = evaluate(model, val_loader)

    train_losses.append(running_loss / len(train_loader))
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Output statistics for each epoch
    # Выводим статистику для каждой эпохи
    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss / len(train_loader)}, "
          f"Validation Accuracy: {val_accuracy:.2f}%, Validation Loss: {val_loss:.4f}")


plt.figure(figsize=(12, 5))

# Losses | Потери
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()

# Accuracy | Точность
plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Val Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Получение всех предсказаний и меток
all_preds = []
all_labels = []

model.eval()
with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.numpy())
        all_labels.extend(labels.numpy())

# Confusion matrix building
# Построение confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Visualization
# Визуализация
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.xticks(rotation=45)
plt.yticks(rotation=45)
plt.tight_layout()
plt.show()

# Saving only the model parameters (weights and offsets)
# Сохраняем только параметры модели (веса и смещения)
torch.save(model.state_dict(), 'GeoClassCNNModel.pth')

# Saving the entire model (not recommended, as it may be due to incompatibility of PyTorch versions)
# Сохраняем всю модель целиком (не рекомендуется, так как может быть связано с несовместимостью версий PyTorch)
torch.save(model, 'GeoClassCNNModel_full.pth')