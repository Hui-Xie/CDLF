% read origin and adversarial data files and display in camparison
%file = '/home/hxie1/temp_advData/20181109_1739/8.txt';
clear all;
clc;
label = 6;
advDir = '/home/hxie1/temp_advData';
targetVec = [0,1,2,3,4,5,6,7,8,9];
originfile = strcat(advDir, '/',num2str(label),'.txt'); %originFile
A = dlmread(originfile); % origin Array;
I = mat2gray(A,[0,255]); % origin image
targetVec(label+1)= [];
subplot(2,5,1); imshow(I); title(strcat('origin',32,num2str(label))); % 32 indicate space
for i=1:9  % target index
   target = targetVec(i);
   targetFile = strcat(advDir, '/',num2str(label),'-Ad', num2str(target),'.txt');
   targetA = dlmread(targetFile);
   targetI = mat2gray(targetA, [0,255]);
   subplot(2,5,i+1); imshow(targetI);title(strcat('target to',32, num2str(target)));
end