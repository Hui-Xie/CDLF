//
// Created by Hui Xie on 3/11/19.
// Copyright (c) 2018 Hui Xie. All rights reserved.

//

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIdentityTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "FileTools.h"
#include "Tools.h"

typedef short PixelType;
const int Dimension =3;
using ImageType = itk::Image< PixelType, Dimension >;

using namespace std;

void printUsage(char* argv0){
    cout<<"============= Interpolate images in a directory ==========="<<endl;
    cout<<"This program uniforms and converts all images in given directory into a same spacing size."<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" <inputImageDir>  <spacingSize> <interpolationMethod> <outputImageDir>"<<endl;
    cout<<"spacingSize: e.g.  1.2*1.2*2  in original image axis order. "<<endl;
    cout<<"interpolationMethod:  0 NearestNeighbor; 1 Linear Spline; 3 Cubic Bspline;" <<endl;
    cout<<endl;
}

int main(int argc, char *argv[]) {
    if (5 != argc )  {
        cout << "Error: number of parameters error." << endl;
        printUsage(argv[0]);
        return -1;
    }

    const string inputDir = argv[1];
    const string spacingSizeStr = argv[2];
    const int interpolationMethod = atoi(argv[3]);
    const string outputDir = argv[4];

    const vector<float> newSpacingVec = str2FloatVector(spacingSizeStr);

    vector<string> imagesVector;
    getFileVector(inputDir, imagesVector);
    int numFiles = imagesVector.size();
    cout<<"Totally read "<<numFiles <<"  image files in directory "<<inputDir<<endl;
    for (int i=0;i<numFiles; ++i){
        const string inputImagePath = imagesVector[i];
        const string filename = getFileName(inputImagePath);
        const string outpuImagePath = outputDir+ "/"+ filename;

        using ReaderType = itk::ImageFileReader< ImageType >;
        typename ReaderType::Pointer reader = ReaderType::New();
        reader->SetFileName( inputImagePath );
        reader->Update();
        typename ImageType::Pointer image = reader->GetOutput();

        // get old Image coordinates
        typename ImageType::RegionType region = image->GetLargestPossibleRegion();
        ImageType::SizeType oldImageSize =region.GetSize();
        ImageType::PointType origin = image->GetOrigin();
        ImageType::SpacingType oldSpacing = image->GetSpacing();
        ImageType::DirectionType direction = image->GetDirection();

        //set new image coordinates
        ImageType::SpacingType newSpacing;
        ImageType::SizeType newImageSize;
        for (int i=0; i<Dimension; ++i){
            newSpacing[i] = newSpacingVec[i];
            newImageSize[i] = oldImageSize[i] * ((float)oldSpacing[i])/newSpacing[i];
        }

        typedef itk::IdentityTransform<double, Dimension> Transform;
        Transform::Pointer transform = Transform::New();
        transform->SetIdentity();

        typedef itk::BSplineInterpolateImageFunction<ImageType, double, double > Interpolator;
        Interpolator::Pointer interpolator = Interpolator::New();
        interpolator->SetSplineOrder(interpolationMethod);

        typedef itk::ResampleImageFilter<ImageType, ImageType> ResampleFilter;
        ResampleFilter::Pointer resampleFilter = ResampleFilter::New();
        resampleFilter->SetTransform(transform);
        resampleFilter->SetInterpolator(interpolator);
        resampleFilter->SetOutputOrigin(origin);
        resampleFilter->SetOutputSpacing(newSpacing);
        resampleFilter->SetSize(newImageSize);
        resampleFilter->SetInput(image);

        // Write the result

        typedef itk::ImageFileWriter<ImageType> WriterType;
        typename WriterType::Pointer writer = WriterType::New();
        writer->SetFileName(outpuImagePath);
        writer->SetInput(resampleFilter->GetOutput());
        writer->Update();

        break; // For single image process for test
    }

    cout<<"All converted files have been output to "<<outputDir<<endl;
    return 0;
}
