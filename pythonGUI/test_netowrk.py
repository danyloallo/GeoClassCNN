import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import torch
from torchvision import transforms
from PIL import Image

class ImageLoader(QWidget):
    def __init__(self):
        super().__init__()

        # Инициализация компонентов
        self.initUI()

    def initUI(self):
        # Creating buttons and controls
        # Создаем кнопки и элементы управления
        self.imageLabel = QLabel(self)  # The place to display the image | Место для вывода изображения
        self.loadButton = QPushButton('Загрузить картинку', self)  # The button for uploading an image | Кнопка для загрузки изображения
        self.predictionLabel = QLabel('Предсказание: Нет данных', self)  # A place to make a prediction | Место для вывода предсказания

        # Setting up a button to upload an image
        # Настройка кнопки для загрузки изображения
        self.loadButton.clicked.connect(self.loadImage)

        # Vertical placement of elements
        # Вертикальное размещение элементов
        layout = QVBoxLayout()
        layout.addWidget(self.loadButton)
        layout.addWidget(self.imageLabel)
        layout.addWidget(self.predictionLabel)

        self.setLayout(layout)

        # Setting up the window
        # Настройка окна
        self.setWindowTitle('Загрузчик изображений')
        self.setGeometry(100, 100, 800, 600)

    def loadImage(self):
        # Open the dialog to select a file
        # Открываем диалог для выбора файла
        options = QFileDialog.Options()
        filePath, _ = QFileDialog.getOpenFileName(self, "Выберите картинку", "", "Image Files (*.png *.jpg *.bmp)",
                                                  options=options)
        if filePath:
            # Upload an image and display
            # Загружаем картинку и отображаем
            pixmap = QPixmap(filePath)
            self.imageLabel.setPixmap(pixmap.scaled(self.imageLabel.size(), Qt.KeepAspectRatio))

            # Running the image through the model
            # Прогоняем картинку через модель
            prediction = self.predict(filePath)
            self.predictionLabel.setText(f'Предсказание: {prediction}')

    def predict(self, image_path):
        # Opening an image using PIL
        # Открытие изображения с помощью PIL
        img = Image.open(image_path).convert('RGB')

        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        img_tensor = transform(img).unsqueeze(0)  # Добавляем дополнительную размерность для батча

        # Loading the model
        # Загружаем модель
        model = torch.load('GeoClassCNNModel_full.pth', weights_only=False)
        model.eval()  # Switching to evaluation mode | Переводим в режим оценки
        class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
                       'Industrial', 'Pasture', 'PermanentCrop', 'Residential', 'River',
                       'SeaLake']
        print('model')

        # Getting predictions
        # Получаем предсказания
        with torch.no_grad():
            outputs = model(img_tensor)
            _, predicted = torch.max(outputs, 1)

        # Getting the class label back
        # Вернем класс
        return class_names[predicted.item()]


if __name__ == '__main__':
    app = QApplication(sys.argv)

    ex = ImageLoader()
    ex.show()
    sys.exit(app.exec_())
