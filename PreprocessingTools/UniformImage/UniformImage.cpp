//
// Created by Hui Xie on 9/6/18.
//
#include <iostream>
#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkRegionOfInterestImageFilter.h"


using namespace std;


void printUsage(char* argv0){
    cout<<"============= Uniform Image in Consistent Size and Spacing ==========="<<endl;
    cout<<"This program uses the input size and spacing to get central volume of input image. "
          "And output image file will be outputed in the parallel ***_uniform directory."<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<"<fullPathFileName> <sizeX> <sizeY> <sizeZ> <spacingX> <spacingY> <spacingZ>"<<endl;
    cout<<endl;
}

string getUniformPathFileName(const string& inputFile){
    string result = inputFile;
    size_t pos = result.rfind('/');
    if (pos != string::npos){
        result.insert(pos, "_uniform");
    }
    else{
        result = "";
    }
    return result;
}

int main(int argc, char *argv[]) {

    printUsage(argv[0]);
    if (argc != 8) {
        cout << "Error: the number of parameters is incorrect." << endl;
        return -1;
    }

    //get input parameter
    const string inputFile = argv[1];
    const int sizeX = atoi(argv[2]);
    const int sizeY = atoi(argv[3]);
    const int sizeZ = atoi(argv[4]);
    const float spacingX = atof(argv[5]);
    const float spacingY = atof(argv[6]);
    const float spacingZ = atof(argv[7]);
    const string outputFile = getUniformPathFileName(inputFile);

    //read input image
    const int Dimension = 3;
    using ImageType = itk::Image< float, Dimension >;
    using ReaderType = itk::ImageFileReader< ImageType >;
    ReaderType::Pointer reader = ReaderType::New();
    reader->SetFileName( inputFile);
    reader->Update();
    ImageType::Pointer image = reader->GetOutput();

    //get input image ROI
    ImageType::SizeType inputSize, outputSize, roiSize;  //roi: region of interest
    ImageType::PointType inputOrigin, outpusOrigin;
    ImageType::SpacingType inputSpacing, outputSpacing;
    ImageType::IndexType start;
    outputSize[0] = sizeX; outputSize[1]=sizeY; outputSize[2]=sizeZ;
    outputSpacing[0] = spacingX; outputSpacing[1]=spacingY; outputSpacing[2]=spacingZ;

    inputSize = image->GetLargestPossibleRegion().GetSize();
    inputOrigin = image->GetOrigin();
    inputSpacing = image->GetSpacing();

    for(int i =0; i<Dimension; ++i){
        roiSize[i] = outputSpacing[i]* outputSize[i]/inputSpacing[i];
        if (roiSize[i] > inputSize[i]){
            cout<<"Error: "<<inputFile<< " has not enough physical volume to support output Size ans spacing at dimension "<<i <<"."<<endl;
            cout<<"Infor: maybe you need to reduce the output Size. "<<endl;
            return -2;
        }
        else{
            start[i] = (inputSize[i] - roiSize[i])/2; // get the central volume of input Image;
        }
    }

    ImageType::RegionType region;
    region.SetIndex(start);
    region.SetSize(roiSize);

    using ROIFilterType = itk::RegionOfInterestImageFilter< ImageType, ImageType >;
    ROIFilterType::Pointer roiFilter = ROIFilterType::New();
    roiFilter->SetInput( image);
    roiFilter->SetRegionOfInterest( region );
    roiFilter->Update();
    image = roiFilter->GetOutput();






    
    

    //set TransformType and resample Type

    //Convert and Write out
    using WriterType = itk::ImageFileWriter< ImageType >;
    WriterType::Pointer writer = WriterType::New();
    // if output does not exist, create


    writer->SetFileName( outputFile);
    writer->SetInput(image);
    writer->Update();

    return 0;
}