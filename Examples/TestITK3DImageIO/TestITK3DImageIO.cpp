//
// Created by Hui Xie on 8/18/2018.
// Copyright (c) 2018 Hui Xie. All rights reserved.
//

#include "ITKImageIO.h"
#include <string>
#include "TestITKDataMgr.h"

void printUsage(char* argv0){
    cout<<"Test ITK 3D image:"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" fullPathInputFileNane fullPathOutputFilename"<<endl;
}


int main(int argc, char *argv[]){
    /*
    if (3 != argc){
        cout<<"Error: input parameter error."<<endl;
        printUsage(argv[0]);
        return -1;
    }

    const string inputFilename = argv[1];
    const string outputFilename = argv[2];

    */
    CPUAttr cpuAttr;
    cpuAttr.getCPUAttr();

#ifdef Use_GPU
    GPUAttr gpuAttr;
    gpuAttr.getGPUAttr();
    cout<<"Info: program use Cuda GPU."<<endl;
#else
    cout<<"Info: program use CPU, instead of GPU."<<endl;
#endif

    /*
     * //Test for basic input and output
     *
    ITKImageIO<float, 3> itkImageIO;

    Tensor<float> *pImage = nullptr;
    itkImageIO.readFile(inputFilename, pImage);

    //change value of pImage,
    vector<int> tensorSize = pImage->getDims();
    vector<int> halfTensorSize = tensorSize / 2;
    for (int i = halfTensorSize[0] - 20; i < halfTensorSize[0] + 20; ++i)
        for (int j = halfTensorSize[1] - 20; j < halfTensorSize[1] + 20; ++j)
            for (int k = halfTensorSize[2] - 20; k < halfTensorSize[2] + 20; ++k) {
                pImage->e(i, j, k) = 0;  //dig a hole in the middle of brain.
            }

    itkImageIO.writeFileWithSameInputDim(pImage, {0, 0, 0}, outputFilename);

    if (nullptr != pImage) {
        delete pImage;
        pImage = nullptr;
    }

     */


    /*cout<<"==================Test Extend label ======================"<<endl;
    string srcFile = "/home/hxie1/temp/HN-CHUM-040_CT_HN.nrrd";
    string smallLabelFile = "/home/hxie1/temp/HN-CHUM-040_GTV.nrrd";
    string bigLabelFile ="/home/hxie1/temp/HN-CHUM-040_GTV_big.nrrd";

    ITKImageIO<int, 3> itkImageIOIntensity;
    ITKImageIO<unsigned char, 3> itkImageIOLabel;
    Tensor<int>* pImage2 = nullptr;
    Tensor<int>* pLabelImage = nullptr;
    itkImageIOIntensity.readFile(srcFile, pImage2);
    itkImageIOIntensity.extendLabelFileVolume(smallLabelFile, pLabelImage);
    itkImageIOIntensity.writeFileWithSameInputDim(pLabelImage, {0,0,0}, bigLabelFile);



    if (nullptr != pImage2) {
        delete pImage2;
        pImage2 = nullptr;
    }

    if (nullptr != pLabelImage) {
        delete pLabelImage;
        pLabelImage = nullptr;
    }*/

    // read a label file , crop it and save it
    const string imageFilePath = "/home/hxie1/data/HeadNeckSCC/ExtractData/CT_Images/HNSCC-01-0017_CT.nrrd";
    const string labelFilePath = "/home/hxie1/data/HeadNeckSCC/ExtractData/GTV_Images/HNSCC-01-0017_GTV.nrrd"; // 512*512*130
    const string outputLabelFilePath = "/home/hxie1/data/HeadNeckSCC/ExtractData/GTV_Images/HNSCC-01-0017_GTV_test.nrrd";
    const string outputCTFilePath =  "/home/hxie1/data/HeadNeckSCC/ExtractData/CT_Images/HNSCC-01-0017_CT_ROI.nrrd";
    TestITKDataMgr dataMgr("");

    // test image clip according to inputSize
    /*
  Tensor<float> *pImage = nullptr;

  dataMgr.readLabelFile(labelFilePath, pImage); // pImage with size: 130*512*512

  Tensor<float> *pSubLabel = new Tensor<float>({90,500,500});
  pImage->subTensorFromTopLeft((pImage->getDims() - pSubLabel->getDims()) / 2, pSubLabel, {1,1,1});

  Tensor<unsigned char> outputLabel({90,500,500});
  outputLabel.valueTypeConvertFrom(*pSubLabel);

  dataMgr.saveLabel2File(&outputLabel, {20,6,6}, outputLabelFilePath);

  if (nullptr != pImage) {
        delete pImage;
        pImage = nullptr;
    }
    if (nullptr != pSubLabel) {
        delete pSubLabel;
        pSubLabel = nullptr;
    }
   */


    // test ROI clip with center of GTV

    Tensor<float> *pLabel = nullptr;
    dataMgr.readLabelFile(labelFilePath, pLabel); // pLabel with size: 130*512*512
    vector<int> center = pLabel->getCenterOfNonZeroElements();
    if (nullptr != pLabel) {
        delete pLabel;
        pLabel = nullptr;
    }

    Tensor<float> *pImage = nullptr;
    dataMgr.readImageFile(imageFilePath, pImage); // pImage with size: 130*512*512
    Tensor<float> *pSubImage = new Tensor<float>({65,121, 121});

    // clip from GTV center
    vector<int> topLeft = dataMgr.getTopLeftIndexFrom(pImage->getDims(), pSubImage->getDims(), center);

    // clip from image center
    //vector<int> topLeft = (pImage->getDims()- pSubImage->getDims())/2;

    pImage->subTensorFromTopLeft(topLeft, pSubImage, {1,1,1});

    dataMgr.saveImage2File(pSubImage, topLeft, outputCTFilePath);


    if (nullptr != pImage) {
        delete pImage;
        pImage = nullptr;
    }
    if (nullptr != pSubImage) {
        delete pSubImage;
        pSubImage = nullptr;
    }

    cout << "============= End of TestITKImageIO =============" << endl;
}