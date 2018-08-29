//
// Created by hxie1 on 8/22/18.
//

#include "TIPLIO.h"

//const string inputFilename = "/Users/hxie1/temp/BRATS_001.nii";
//const string outputFilename = "/Users/hxie1/temp/BRATS_001_Output.nii";

void printUsage(char* argv0){
    cout<<"Test TIPL image:"<<endl;
    cout<<"This interface does not support compress image file, eg. gz file"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" fullPathInputFileNane fullPathOutputFilename"<<endl;
}


int main(int argc, char *argv[]){

    if (3 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string inputFilename = argv[1];
    const string outputFilename = argv[2];

    TIPLIO<float, 3> imageIO;

    Tensor<float>* pImage = nullptr;
    int result = imageIO.readNIfTIFile(inputFilename, pImage);
    if (0 != result){
        return -1;
    }

    //change value of pImage,
    vector<long> tensorSize = pImage->getDims();
    vector<long> halfTensorSize = tensorSize /2;
    for(long i=halfTensorSize[0]-20;i<halfTensorSize[0]+20;++i)
        for(long j=halfTensorSize[1]-20; j<halfTensorSize[1]+20;++j)
            for(long k=halfTensorSize[2]-20;k<halfTensorSize[2]+20;++k){
                pImage->e(i,j,k) = 0;  //dig a hole in the middle of brain.
            }

    imageIO.write3DNIfTIFile(pImage, {0,0,0}, outputFilename);

    if (nullptr != pImage){
        delete pImage;
        pImage = nullptr;
    }
    cout<<"================End of TIPL Read Writer==========="<<endl;

    return 0;
}

