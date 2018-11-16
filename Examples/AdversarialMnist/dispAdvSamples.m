% read origin and adversarial data files and display in camparison
clear all;
clc;
label = 1;
advDir = '/home/hxie1/temp_advData';
targetVec = [0,1,2,3,4,5,6,7,8,9];
originfile = strcat(advDir, '/',num2str(label),'.txt'); %originFile
A = dlmread(originfile); % origin Array;
I = mat2gray(A,[0,255]); % origin image
targetVec(label+1)= [];
subplot(2,5,1); imshow(I); title(strcat('Origin',32,num2str(label))); % 32 indicate space
for i=1:9  % target index
   target = targetVec(i);
   targetFile = strcat(advDir, '/',num2str(label),'-Ad', num2str(target),'.txt');
   if exist(targetFile, 'file')== 2
       targetA = dlmread(targetFile);
   else
       targetA = zeros(size(A));
   end
   targetI = mat2gray(targetA, [0,255]);
   
   diffA = targetA - A;
   infNorm = norm(diffA(:), Inf); % where : indiates vectorize 
   infNormStr = sprintf('L_{inf}(d)=%.0f', infNorm);

   corr = corrcoef(A(:), targetA(:));
   corrStr = sprintf('Corr(A,O)=%.4f', corr(1,2));
   
   titleText = {strcat('Target to',32, num2str(target)), infNormStr, corrStr};
   subplot(2,5,i+1); imshow(targetI);title(titleText);
end