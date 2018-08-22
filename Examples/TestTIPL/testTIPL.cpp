//
// Created by hxie1 on 8/22/18.
//

#include "TIPLIO.h"

const string filename = "/Users/hxie1/temp/BRATS_001.nii";


int main(int argc, char *argv[]){

    TIPLIO<unsigned short, 3> imageIO;

    Tensor<float>* pImage = nullptr;
    imageIO.readNIfTIFile(filename, pImage);

    //change value of pImage,
    vector<long> tensorSize = pImage->getDims();
    vector<long> halfTensorSize = tensorSize /2;
    for(long i=halfTensorSize[0]-20;i<halfTensorSize[0]+20;++i)
        for(long j=halfTensorSize[1]-20; j<halfTensorSize[1]+20;++j)
            for(long k=halfTensorSize[2]-20;k<halfTensorSize[2]+20;++k){
                pImage->e(i,j,k) = 0;  //dig a hole in the middle of brain.
            }

    imageIO.write3DNIfTIFile(pImage, {0,0,0}, "//Users/hxie1/temp/BRATS_001_Output.nii");

    if (nullptr != pImage){
        delete pImage;
        pImage = nullptr;
    }
    cout<<"End of TIPL Read Writer"<<endl;


}

