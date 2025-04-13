load('GeoClassCNN.mat')
[filename, filepath] = uigetfile({'*.jpg;*.png;*.jpeg', 'Image Files (*.jpg, *.png, *.jpeg)'}, ...
                                 'Выберите изображение');
if isequal(filename, 0)
    disp('Выбор файла отменён.');
else
    % Полный путь к выбранному файлу
    imgPath = fullfile(filepath, filename);

    % Загрузка и обработка изображения
    img = imread(imgPath);
    imgResized = imresize(img, [64 64]);  % Изменение размера для сети

    % Предсказание класса
    predictedLabel = classify(net, imgResized);

    % Отображение результата
    figure;
    imshow(img);
    title(['Predicted: ' char(predictedLabel)]);
end
