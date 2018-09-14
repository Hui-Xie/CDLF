//
// Created by Hui Xie on 9/8/18.
//
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkLinearInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkStatisticsImageFilter.h"
#include "itkImageRegionIterator.h"
#include "FileTools.h"


using namespace std;


void printUsage(char* argv0){
    cout<<"============= Uniform Image in Consistent Size and Spacing ==========="<<endl;
    cout<<"The basic idea is to use same physical-size volumes as input for deep learning network."<<endl;
    cout<<"This program uses the input size and spacing to get central volume of input image. "
          "And output image file will be outputed in the parallel ***_uniform directory."<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<fullPathFileName> <pathSuffix> <labelChange> <sizeX> <sizeY> <sizeZ> <spacingX> <spacingY> <spacingZ>"<<endl;
    cout<<"Paraemeters Notes:"<<endl
        <<"pathSuffix: used to generate outputfile's path suffix"<<endl
        <<"labelChange: 0: no changes; 2To1: lable 2 changes to 1; 3To0: label 3 changes to 0, etc"<<endl;
    cout<<endl;
}



int main(int argc, char *argv[]) {

    printUsage(argv[0]);
    if (argc != 10) {
        cout << "Error: the number of parameters is incorrect." << endl;
        return -1;
    }

    //get input parameter
    const string inputFile = argv[1];
    const string pathSuffix = argv[2];
    const string labelChange = argv[3];
    const int sizeX = atoi(argv[4]);
    const int sizeY = atoi(argv[5]);
    const int sizeZ = atoi(argv[6]);
    const float spacingX = atof(argv[7]);
    const float spacingY = atof(argv[8]);
    const float spacingZ = atof(argv[9]);
    const string outputFile = getUniformPathFileName(inputFile, pathSuffix+"_"+labelChange);

    //read input image
    const int Dimension = 3;
    using ImageType = itk::Image< unsigned char, Dimension >;
    using ReaderType = itk::ImageFileReader< ImageType >;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( inputFile);
    reader->Update();
    ImageType::Pointer image = reader->GetOutput();

    //Get Image min pixel value for resample default value
    typedef itk::StatisticsImageFilter<ImageType> StatisticsImageFilterType;
    StatisticsImageFilterType::Pointer statisticsImageFilter = StatisticsImageFilterType::New ();
    statisticsImageFilter->SetInput(image);
    statisticsImageFilter->Update();
    ImageType::ValueType  minPixelValue = statisticsImageFilter->GetMinimum();

    //get input image ROI
    ImageType::SizeType inputSize, outputSize, roiSize;  //roi: region of interest
    ImageType::PointType inputOrigin, outputOrigin;
    ImageType::SpacingType inputSpacing, outputSpacing;
    ImageType::DirectionType inputDirection;
    ImageType::IndexType start;
    outputSize[0] = sizeX; outputSize[1]=sizeY; outputSize[2]=sizeZ;
    outputSpacing[0] = spacingX; outputSpacing[1]=spacingY; outputSpacing[2]=spacingZ;

    inputSize = image->GetLargestPossibleRegion().GetSize();
    inputOrigin = image->GetOrigin();
    inputSpacing = image->GetSpacing();
    inputDirection = image->GetDirection();

    for(int i =0; i<Dimension; ++i){
        float redundancy = inputSize[i]*inputSpacing[i] - outputSize[i]*outputSpacing[i];
        if (redundancy < 0){
            cout<<"Error: the image "<<inputFile <<" is not big enough to suppport the output spacing and size at dimension "<< i<<"."<<endl;
            return -1;
        }
        else{
            start[i] = redundancy/(2*outputSpacing[i]);// this index is in the measurement of after resampled image
        }
        outputOrigin[i] = inputOrigin[i] + start[i]*outputSpacing[i]*inputDirection[i][i];
    }

    using LinearInterpolatorType = itk::LinearInterpolateImageFunction< ImageType>;
    LinearInterpolatorType::Pointer interpolator = LinearInterpolatorType::New();

    using ResampleFilterType = itk::ResampleImageFilter< ImageType, ImageType >;
    ResampleFilterType::Pointer resampleFilter = ResampleFilterType::New();
    resampleFilter->SetInput( image );
    resampleFilter->SetInterpolator( interpolator );
    resampleFilter->SetSize( outputSize );
    resampleFilter->SetOutputSpacing( outputSpacing);
    resampleFilter->SetOutputOrigin( outputOrigin );
    resampleFilter->SetOutputDirection(inputDirection);
    resampleFilter->SetDefaultPixelValue(minPixelValue);
    resampleFilter->Update();
    image = resampleFilter->GetOutput();

    // convert label
    if ("0" != labelChange){
        itk::ImageRegionIterator<ImageType> iter(image,image->GetLargestPossibleRegion());
        iter.GoToBegin();
        while(!iter.IsAtEnd())
        {
            if (1 == iter.Get() && labelChange == "1To0" ){
                iter.Set(0);
            }
            if (2 == iter.Get() && labelChange == "2To1"){
                iter.Set(1);
            }
            if (3 == iter.Get() && labelChange == "3To1"){
                iter.Set(1);
            }
            if (4 == iter.Get() && labelChange == "4To1"){
                iter.Set(1);
            }
            ++iter;
        }
    }

    //Convert and Write out
    string outputDir = getDirFromFileName(outputFile);
    if (!dirExist(outputDir)){
        mkdir(outputDir.c_str(),S_IRWXU |S_IRWXG | S_IROTH |S_IXOTH);
    }
    using WriterType = itk::ImageFileWriter< ImageType >;
    WriterType::Pointer writer = WriterType::New();
    writer->SetFileName( outputFile);
    writer->SetInput(image);
    writer->Update();
    cout<<"Info: "<< outputFile <<" outputed. "<<endl;

    return 0;
}
