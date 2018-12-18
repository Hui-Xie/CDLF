% read txt image files and display
clear all;
clc;
label = 2;
dir = '/home/hxie1/temp_DecoderOutput';
originFile = strcat(dir, '/T','1-I4','.txt'); %origin input File
reconstrFile = strcat(dir, '/T','1-R4','.txt'); %Reconstruction file
O = dlmread(originFile); R = dlmread(reconstrFile);
OImage = mat2gray(O,[0,255]); RImage = mat2gray(R,[0,255]);

[filepath,nameO,ext] = fileparts(originFile);
subplot(1,2,1); imshow(OImage);title(nameO);

[filepath,nameR,ext] = fileparts(reconstrFile);
subplot(1,2,2); imshow(RImage);title(nameR);