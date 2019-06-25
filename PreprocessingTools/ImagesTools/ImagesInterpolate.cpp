//
// Created by Hui Xie on 3/11/19.
// Copyright (c) 2019 Hui Xie. All rights reserved.

//

#include "itkImageFileReader.h"
#include "itkImageFileWriter.h"
#include "itkIdentityTransform.h"
#include "itkBSplineInterpolateImageFunction.h"
#include "itkResampleImageFilter.h"
#include "itkImageSliceConstIteratorWithIndex.h"
#include "itkImageSliceIteratorWithIndex.h"
#include "FileTools.h"
#include "Tools.h"
#include <algorithm>

typedef short PixelType;
const int Dimension =3;
using ImageType = itk::Image< PixelType, Dimension >;

using namespace std;

void printUsage(char* argv0){
    cout<<"============= Interpolate images in a directory ==========="<<endl;
    cout<<"This program uniforms and converts all images in given directory into a same spacing size."<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" <inputImageDir>  <spacingSize> <interpolationMethod> <outputImageDir> <weakAnnotation>"<<endl;
    cout<<"spacingSize: e.g.  1.2*1.2*2  in original image axis order. "<<endl;
    cout<<"interpolationMethod:  0 NearestNeighbor; 1 Linear Spline; 3 Cubic Bspline;" <<endl;
    cout<<"weakAnnotation:  1 weakAnnotation; 0 full label;" <<endl;
    cout<<"When we discuss the NearestNeighbor interpolation for label images:\n"
          "1  If the label image is continuously labeled, both the bigger spacing or smaller spacing interpolation does not bring hole label;\n"
          "2  If the label image is weakly annotated, for example, in the z axis (from inferior to superior direction) the label is sparsely labeled.\n"
          "    2A. when original image spacing >= target image spacing and orgin coordinates keep unchange, label informatino does not lose.\n"
          "    2B. when original image spacing < target image spacing, label information may lose because of interpolation and sparcity.\n"
          "        In this 2B case in order to avoid this kind of label information lossing, \n"
          "        we need to consider to disperse the label in the original image to its left and right(up and down) neighbors within a radius of 1/2 target image spacing.\n"
         <<endl;
    cout<<"In medical weak annotation context, generally the some of transverse planes are full labeled inside this plane, and some are left out without labels.\n"
          "In this case, when original image spacing < target image spacing, we need to consider to disper the labels along z direction.\n"
        <<endl;

    cout<<"We normaly convert all images into uniformal minimal spacing, and then use them as a start point to further interpoloation."<<endl;

    cout<<"we currently only support z directon dispersing."<<endl;

    cout<<endl;
}

int main(int argc, char *argv[]) {
    if (6 != argc )  {
        cout << "Error: number of parameters error." << endl;
        printUsage(argv[0]);
        return -1;
    }

    const string inputDir = argv[1];
    const string spacingSizeStr = argv[2];
    const int interpolationMethod = atoi(argv[3]);
    const string outputDir = argv[4];
    const int weakAnnotation = atoi(argv[5]);

    const vector<float> newSpacingVec = str2FloatVector(spacingSizeStr);

    vector<string> imagesVector;
    getFileVector(inputDir, imagesVector);
    const int numFiles = imagesVector.size();
    cout<<"Totally read "<<numFiles <<"  image files in directory "<<inputDir<<endl;
    for (int i=0;i<numFiles; ++i){
        const string inputImagePath = imagesVector[i];
        const string filename = getFileName(inputImagePath);
        const string outpuImagePath = outputDir+ "/"+ filename;

        if (fileExist(outpuImagePath)) {
            continue;
        }

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
        for (int j=0; j<Dimension; ++j){
            newSpacing[j] = newSpacingVec[j];
            newImageSize[j] = oldImageSize[j] * ((float)oldSpacing[j])/newSpacing[j];
        }

        if (newSpacing == oldSpacing){
            string cmdStr = string("cp ") + inputImagePath + " " +  outpuImagePath;
            int result = system(cmdStr.c_str());
            if (0 != result){
                cout<<cmdStr << ": runs error"<<endl;
            }
        }
        else{
            //disperse the label in the z direction
            if (1 == weakAnnotation && 0 == interpolationMethod && newSpacing[2] > oldSpacing[2]){
                // get radius for dispersing
                const int radius = int(newSpacing[2]*1.0/oldSpacing[2] +0.5);

                // get z index of all labeled slices
                vector<int> labelledZIndexVector;
                itk::ImageSliceConstIteratorWithIndex<ImageType> sliceIt(image, region);
                sliceIt.GoToBegin();
                int sliceIndex = 0;
                while( !sliceIt.IsAtEnd() )
                {
                    bool labelled = false;
                    while ( !sliceIt.IsAtEndOfSlice() )
                    {
                        while ( !sliceIt.IsAtEndOfLine() )
                        {
                            if (sliceIt.Get() >0) {
                               labelled = true;
                               break;
                            }
                            else{
                                ++sliceIt;
                            }
                        }
                        if (labelled) {
                            break;
                        }
                        else{
                            sliceIt.NextLine();
                        }
                    }
                    if (labelled){
                        labelledZIndexVector.push_back(sliceIndex)
                    }
                    ++sliceIndex;
                    sliceIt.NextSlice();
                }
                // disperse labelled slice within radius
                int Len = labelledZIndexVector.size();
                for(int n=0; n< Len; ++n){
                    int sliceL = labelledZIndexVector[n];
                    int zLow = std::max(sliceL -radius, 0);
                    if (n>0 && zLow <= labelledZIndexVector[n-1] + radius){
                        zLow = (labelledZIndexVector[n-1] + labelledZIndexVector[n])/2;
                    }
                    int zHigh = std::min(sliceL + radius, int(oldImageSize[2]-1));
                    if (n<Len-1 && zHigh >= labelledZIndexVector[n+1]-radius){
                        zHigh = (labelledZIndexVector[n] + labelledZIndexVector[n+1])/2;
                    }

                    for(int x=0; x < oldImageSize[0]; ++x)
                        for (int y =0; y<oldImageSize[1];++y){
                            const ImageType::IndexType labelIndex = {{x,y,sliceL}};    // Position of {X,Y,Z}
                            PixelType value = image->GetPixel(labelIndex);
                            for (int z= zLow; z<=zHigh;  ++z ){
                                if (z == sliceL){
                                    continue;
                                }
                                else{
                                    const ImageType::IndexType disperseIndex = {{x,y,z}};
                                    image->SetPixel(disperseIndex, value);
                                }
                            }
                        }
                }
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

            resampleFilter->SetOutputDirection(direction);
            resampleFilter->SetInput(image);


            // Write the result

            typedef itk::ImageFileWriter<ImageType> WriterType;
            typename WriterType::Pointer writer = WriterType::New();
            writer->SetFileName(outpuImagePath);
            writer->SetInput(resampleFilter->GetOutput());
            writer->Update();
        }
    }

    cout<<"All converted files have been output to "<<outputDir<<endl;
    return 0;
}
