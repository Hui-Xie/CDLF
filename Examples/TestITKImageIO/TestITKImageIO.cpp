//
// Created by Hui Xie on 8/18/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ITKImageIO.h"
#include <string>

const string filename = "/E/CProject/Images/Data/1103/3/NIFTI/1103_3.nii";

int main (int argc, char *argv[])
{
   ITKImageIO<unsigned short, 3> itkImageIO;

   Tensor<float>* pImage = nullptr;
   itkImageIO.readFile(filename, pImage);

   //change value of pImage,original image 256*256*332
   for(long i=110;i<150;++i)
       for(long j=110; j<150;++j)
           for(long k=140;j<180;++k){
               pImage->e(i,j,k) = 0;  //dig a hole in the middle of brain.
           }

   itkImageIO.writeFile(pImage, {0,0,0}, "itkOutput.nii");

   if (nullptr != pImage){
      delete pImage;
      pImage = nullptr;
   }
   cout<<"End of ITK Read Writer"<<endl;
}