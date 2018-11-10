% display the gradient evoluation
clear all;
clc;
label = 6;
target = 5;
NGradient = 14; % 0-NGradient-1
advDir = '/home/hxie1/temp_advData';

figure;
% display original file
originfile = strcat(advDir, '/',num2str(label),'.txt'); %originFile
A = dlmread(originfile); % origin Array;
I = mat2gray(A,[0,255]); % origin image
subplot(1,2,1); imshow(I); title(strcat('origin',32,num2str(label))); % 32 indicate space

% display target file
targetFile = strcat(advDir, '/',num2str(label),'-Ad', num2str(target),'.txt');
targetA = dlmread(targetFile);
targetI = mat2gray(targetA, [0,255]);
subplot(1,2,2); imshow(targetI);title(strcat('target to',32, num2str(target)));

% display gradient files
figure;
row = 3; col = ceil(NGradient/row);
for i=0:NGradient-1  % gradient index
   gFile = strcat(advDir, '/',num2str(label),'-Ad', num2str(target),'-G',num2str(i),'.txt'); %gradient file
   gA = dlmread(gFile);
   % gray gradient image
   gI = mat2gray(gA);
   subplot(row,col,i+1); imshow(gI);title(strcat('iteration',32, num2str(i)));
   
   % color gradient image
   %subplot(row,col,i+1); image(gA.*100000);colorbar; title(strcat('iteration',32, num2str(i)));
   
end
disp("Color gradient image is enlarge 100000 times.");

