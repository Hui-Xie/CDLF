//
// Created by Hui Xie on 8/18/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ITK_RWriter.h"
#include <string>

const string filename = "/E/CProject/Images/Data/1103/3/NIFTI/1103_3.nii";

int main (int argc, char *argv[])
{
   ITK_RWriter<unsigned short, 3> readWriter;

   Tensor<float>* pImage = nullptr;

   readWriter.readFile(filename, pImage);

   cout<<"End of ITK Read Writer"<<endl;
}