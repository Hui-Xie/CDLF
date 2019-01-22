% read original image, adversarial image,  reconstructed image and display
% and compute their correlation coefficient 
clear all;
clc;

% for reconstruction from adversarial smaples
dir = '/home/hxie1/temp_advData';
originFile = strcat(dir, '/6','.txt'); %origin input File
advFile = strcat(dir, '/6-Ad4','.txt'); %adversaril File
reconstrFile = strcat(dir, '/6-Ad4-R4','.txt'); %Reconstruction file

origArray = dlmread(originFile); advArray = dlmread(advFile); reArray = dlmread(reconstrFile);
origImage = mat2gray(origArray,[0,255]); advImage = mat2gray(advArray, [0,255]); ReImage = mat2gray(reArray,[0,255]);

corrMatrix = corrcoef(reshape(origArray,784,1), reshape(reArray, 784, 1));
corr_O_R = corrMatrix(1,2);

corrMatrix = corrcoef(reshape(advArray,784,1), reshape(reArray, 784, 1));
corr_A_R = corrMatrix(1,2);

displayText_O_R = sprintf('\nCorrCoef_O_R=%f', corr_O_R);
displayText_A_R = sprintf('\nCorrCoef_A_R=%f', corr_A_R);

[filepath,nameOrig,ext] = fileparts(originFile);
subplot(1,3,1); imshow(origImage, []);title(strcat('origImage: ',nameOrig, displayText_O_R));

[filepath,nameAdv,ext] = fileparts(advFile);
subplot(1,3,2); imshow(advImage, []);title(strcat('adverImage: ',nameAdv, displayText_A_R));

[filepath,nameRe,ext] = fileparts(reconstrFile);
subplot(1,3,3); imshow(ReImage, []);title(strcat('ReconstImage: ', nameRe));


% sgtitle(displayText);
