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

subplot(1,4,1); imshow(Y);title('Y');
subplot(1,4,2); imshow(dY);title('dY');
subplot(1,4,3); imshow(reconstructGT);title('reconstructed GT');
subplot(1,4,4); imshow(gT);title('the real GT');