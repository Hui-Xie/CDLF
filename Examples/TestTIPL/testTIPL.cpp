//
// Created by hxie1 on 8/22/18.
//

#include "TIPLIO.h"

const string filename = "~/temp/BRATS_001.nii";


int main(int argc, char *argv[]){

    TIPLIO<unsigned short, 3> imageIO;

    Tensor<float>* pImage = nullptr;
    imageIO.readNIfTIFile(filename, pImage);

    //change value of pImage,original image 256*256*332
    for(long i=110;i<150;++i)
        for(long j=110; j<150;++j)
            for(long k=140;j<180;++k){
                pImage->e(i,j,k) = 0;  //dig a hole in the middle of brain.
            }

    imageIO.writeNIfTIFile(pImage, {0,0,0}, "~/temp/BRATS_001_Output.nii");

    if (nullptr != pImage){
        delete pImage;
        pImage = nullptr;
    }
    cout<<"End of ITK Read Writer"<<endl;


}

