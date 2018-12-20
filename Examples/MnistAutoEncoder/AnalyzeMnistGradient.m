% display Gradient and GroudTruth
clear all;
clc;
dir = '/home/hxie1/temp_netParameters/MnistAutoEncoder';
dYFile = strcat(dir, '/dY_100.csv'); 
YFile = strcat(dir, '/Y_100.csv'); 
gtFile = strcat(dir, '/GroundTruth.csv'); 
dY = reshape(dlmread(dYFile), 28,28);
Y = reshape(dlmread(YFile),28, 28);
gT = reshape(dlmread(gtFile),28, 28);

reconstructGT = Y -dY;

subplot(1,4,1); imshow(Y, []);title('Y');
subplot(1,4,2); imshow(dY,[]);title('dY');
subplot(1,4,3); imshow(reconstructGT,[]);title('reconstructed GT');
subplot(1,4,4); imshow(gT,[]);title('the real GT');

% display the k -l layer
figure;
layerVec =[50, 60, 62, 70, 72, 80, 90, 92, 100];
N = length(layerVec);
for i= 1:N
    fileY = strcat(dir, '/Y_', num2str(layerVec(i)),'.csv');
    filedY = strcat(dir, '/dY_', num2str(layerVec(i)),'.csv');
    dataY = dlmread(fileY);
    datadY = dlmread(filedY);
    edgeY = sqrt(length(dataY));
    edgedY = sqrt(length(datadY));
    dataY = reshape(dataY, edgeY,edgeY);
    datadY = reshape(datadY, edgedY,edgedY);
    nameY = strcat('Y', num2str(layerVec(i)));
    namedY = strcat('dY', num2str(layerVec(i)));
    subplot(2,N, i);imshow(dataY, []);title(nameY);
    subplot(2,N, i+N);imshow(datadY, []);title(namedY);
end
