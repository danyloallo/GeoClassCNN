# 🇷🇺 Russian Version 
## Проект по классификации изображений EuroSAT
### 🔧 Реализации 
Это простой проект, который состоит из трёх реализаций решения для классификации спутниковых изображений из датасета EuroSAT. Реализации следующие:
### 1️⃣ Реализация на MATLAB
   Рекомендуется использовать матлаб не ниже версии 2021b, либо как минимум 2018b. В папке находится 3 файла.  
   ```train_model.m``` - файл в котором можно запустить обучение и самим понаблюдать за графиком обучения (обратите внимание, что датасет должен быть на вашем компьютере и в коде нужно указать корректный путь к датасету).  
   ```test_model.m``` - файл в котором при запуске вы выбираете фото и вам выдает предсказанный класс поверхности.  
   ```GeoClassCNN.mat``` - обученная модель, которая может использоваться для теста.  
   В этой части я оставил мало комментариев и то на русском, не думаю что эта часть сильно интересна. 
### 2️⃣ Jupyter Notebook (рекомендуется использовать Google Collab)
   Рекомендуется использовать Google Collab, так как он не будет нагружать вашу систему, а все вычисления производятся на сервере. Проект разбит на 3 части. Первая часть это скачивание и подготовка данных. Вторая часть это настройка и обучение модели. Третья часть это тестирование модели. Здесь очень подробно объясняется каждый шаг, а так же прокоментированы почти все строчки.
### 3️⃣ Реализация с GUI интерфейсом
   По сути я просто скопировал все нужные блоки кода из ```.ipynb``` файла, а так же добавил реализацию с простым интерфейсом, где в окне вы просто выбираете картинку и вам высвечивается сама картинка и результат предсказания. Комментарии кода такие же как и в предыдущем пункте.
#### Рекомендуется использовать вторую реализацию с Google Collab, так как это проще и быстрее. 
Так же в проекте есть папка ```test_images``` с 5 картинками для теста работы модели. Вы можете сами используя Google Maps или Яндекс Карты сделать снимок поверхности в режиме спутника и проверить работу модели (обратите внимание, что масштаб лучше использовать как в картинках в папке ```test_images```).
### 📂 Датасет
В этом проекте используется датасет EuroSAT, который можно скачать по ссылке: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset.  
В этом датасете 10 типов поверхности земли:
1.	🏘 Residential — Жилая застройка
2.	🏭 Industrial — Промышленные объекты
3.	🌊 River — Реки
4.	🛣 Highway — Автомобильные дороги
5.	🐄 Pasture — Пастбища
6.	🌾 PermanentCrop — Постоянные сельскохозяйственные культуры
7.	🌅 SeaLake — Моря и озёра
8.	🌿 HerbaceousVegetation — Травянистая растительность
9.	🌲 Forest — Лес
10.	🌱 AnnualCrop — Сезонные сельскохозяйственные культуры
  
Так же помимо папок с изображениям в нем 3 csv файла, которые разбивают датасет на выборки для обучения, валидации и теста. Датасет состоит из двух частей, в одном изображения формата jpg, в другом tif. Мы работаем с изображениями формата jpg, вторая часть с tif нам не нужна.
### ⚙️ Установка и настройка
▶️ MATLAB
   - Убедитесь, что у вас версия не ниже 2021b (или хотя бы 2018b)
   - Загрузите скрипты
   - В файле ```train_model.m``` измените путь к датасету как на вашем компьютере (переменная ```dataPath```).
   - Скрипты готовы к использованию
  
▶️ Jupyter Notebook (Google Collab)
   - Загрузите файлы для этого проекта 
   - Запустите Google Collab
   - Загрузите файл ```.ipynb```
   - Выполняйте код последовательно блок за блоком
  
▶️ Python GUI
   - Загрузите файлы
   - Используйте комманду ```pip install -r requirements.txt```, которая загрузит все нужные библиотеки для проекта
   - Запустите файл для обучения или для применения модели.

# 🇬🇧 English Version
## EuroSAT Image Classification Project
### 🔧 Implementations
This is a simple project that consists of three different implementations for classifying satellite images from the EuroSAT dataset. The implementations are as follows:
### 1️⃣ MATLAB Implementation
   It is recommended to use MATLAB version 2021b or newer (at least 2018b). The folder contains 3 files.  
   ```train_model.m``` — a file that allows you to start training and observe the training graph (note that the dataset must be located on your computer, and you need to set the correct path to the dataset in the code).  
   ```test_model.m``` — a file where you can select an image and get the predicted surface class.  
   ```GeoClassCNN.mat``` — a pre-trained model that can be used for testing.  
   This part contains few comments, and they are in Russian — I don’t think this part is particularly interesting.

### 2️⃣ Jupyter Notebook (recommended to use Google Colab)
   It is recommended to use Google Colab, as it won’t load your system and all computations are done on the server. The project is split into 3 parts:  
   Downloading and preparing the data.  
   Model configuration and training.  
   Model testing.  
   Each step is explained in great detail, and almost every line is commented.

### 3️⃣ GUI-based Python Implementation
   Essentially, I just copied the necessary code blocks from the .ipynb file and added a simple GUI, where you select an image in a window and get the predicted result along with the image displayed. The code comments are the same as in the previous implementation.
#### It is recommended to use the second implementation with Google Colab, as it is simpler and faster.  
There is also a ```test_images``` folder with 5 images for testing the model’s performance. You can also use Google Maps or Yandex Maps to capture satellite-view screenshots and test the model. (Note: use a zoom level similar to the one in the ```test_images``` folder.)
### 📂 Dataset
This project uses the EuroSAT dataset, which can be downloaded from the following link: https://www.kaggle.com/datasets/apollo2506/eurosat-dataset  
The dataset contains 10 types of land surfaces:  
1.	🏘 Residential 
2.	🏭 Industrial
3.	🌊 River 
4.	🛣 Highway 
5.	🐄 Pasture 
6.	🌾 PermanentCrop 
7.	🌅 SeaLake 
8.	🌿 HerbaceousVegetation
9.	🌲 Forest 
10.	🌱 AnnualCrop

In addition to folders with images, the dataset includes 3 CSV files that split the dataset into training, validation, and test sets. The dataset has two parts: one with JPG images and one with TIF images. We are working with the JPG images; the TIF version is not needed.
### ⚙️ Installation and Setup
▶️ MATLAB
   - Make sure your version is 2021b or newer (at least 2018b)
   - Download the scripts
   - In the ```train_model.m``` file, update the path to the dataset on your computer ( ```dataPath``` variable).
   - The scripts are ready to use
  
▶️ Jupyter Notebook (Google Collab)
   - Download the project files
   - Launch Google Colab
   - Upload the ```.ipynb``` file
   - Run the code block by block in sequence
  
▶️ Python GUI
   - Download the files
   - Use the command ```pip install -r requirements.txt``` to install all necessary libraries
   - Run the training or model inference file
