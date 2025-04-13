clear all
clc
clear

% Путь к папке с датасетом
dataPath = 'C:\Users\bandi\matlabfiles\earthNetwork\ds';

% Чтение данных из CSV
validationData = readtable(strcat(dataPath, '\valid.csv'));
testData = readtable(strcat(dataPath, '\test.csv'));
trainData = readtable(strcat(dataPath, '\train.csv'));

% Построение полных путей к файлам
testData.Filename = fullfile(dataPath, testData.Filename);
trainData.Filename = fullfile(dataPath, trainData.Filename);
validationData.Filename = fullfile(dataPath, validationData.Filename);

% Создание datastore с метками
imdsTrain = imageDatastore(trainData.Filename, ...
    'Labels', categorical(trainData.ClassName));
imdsTest = imageDatastore(testData.Filename, ...
    'Labels', categorical(testData.ClassName));
imdsValidation = imageDatastore(validationData.Filename, ...
    'Labels', categorical(validationData.ClassName));

% Размер входных изображений
inputSize = [64 64 3];

% Аугментация изображений (изменение размера, повороты)
augmentedTrain = augmentedImageDatastore(inputSize, imdsTrain);
augmentedValidation = augmentedImageDatastore(inputSize, imdsValidation);
augmentedTest = augmentedImageDatastore(inputSize, imdsTest);

% Задаем слои нейросети
layers = [
    imageInputLayer(inputSize, 'Normalization', 'zerocenter')

    convolution2dLayer(3, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    convolution2dLayer(3, 64, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer

    maxPooling2dLayer(2, 'Stride', 2)

    fullyConnectedLayer(numel(categories(imdsTrain.Labels)))
    softmaxLayer
    classificationLayer
];

% Задаем параметры обучения
options = trainingOptions('adam', ...
    'MaxEpochs', 15, ...
    'MiniBatchSize', 32, ...
    'ValidationData', augmentedValidation, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'Plots', 'training-progress');

net = trainNetwork(augmentedTrain, layers, options);

% Проверяем точность на тестовой выборке
YPred = classify(net, augmentedTest);
YTest = imdsTest.Labels;

% Расчёт точности
accuracy = sum(YPred == YTest) / numel(YTest);
disp("Test accuracy: " + accuracy);

save('GeoClassCNN.mat', 'net');


