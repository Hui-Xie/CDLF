//
// Created by Hui Xie on 8/29/18.
//

#include "ITKImageIO.h"
#include <string>

void printUsage(char* argv0){
    cout<<"Test ITK 4D image:"<<endl;
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


#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    ITKImageIO<float, 4> itkImageIO;

    Tensor<float> *pImage = nullptr;
    itkImageIO.readFile(inputFilename, pImage);

    string stemName = getStemName(outputFilename);

    int n = pImage->getDims()[0];
    for (int i=0;i<n; ++i){
        string outputName = stemName+"_"+std::to_string(i)+".nii";
        Tensor<float>* pVolume = nullptr;
        pImage->volume(i,pVolume);
        itkImageIO.writeFileWithLessInputDim(pVolume, {0, 0, 0}, outputName);
        if(nullptr != pVolume){
            delete pVolume;
        }
    }

    if (nullptr != pImage) {
        delete pImage;
        pImage = nullptr;
    }
    cout << "============= End of TestITKImageIO =============" << endl;
}
