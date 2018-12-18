% read txt image files and display
% and compute their correlation coefficient 
clear all;
clc;
dir = '/home/hxie1/temp_DecoderOutput';
originFile = strcat(dir, '/T','0-I0','.txt'); %origin input File
reconstrFile = strcat(dir, '/T','0-R0','.txt'); %Reconstruction file
I = dlmread(originFile); R = dlmread(reconstrFile);
IImage = mat2gray(I,[0,255]); RImage = mat2gray(R,[0,255]);

corrMatrix = corrcoef(reshape(I,784,1), reshape(R, 784, 1));
corr = corrMatrix(1,2);

displayText = sprintf('Correlation Coefficient: %f', corr);


[filepath,nameI,ext] = fileparts(originFile);
subplot(1,2,1); imshow(IImage);title(nameI);

[filepath,nameR,ext] = fileparts(reconstrFile);
subplot(1,2,2); imshow(RImage);title(nameR);


% sgtitle(displayText);
disp(displayText);