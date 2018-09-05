//
// Created by hxie1 on 9/5/18.
//

#include <iostream>
#include <vector>
#include <string>
#include <sys/types.h>
#include <dirent.h>
#include "itkImageFileReader.h"

using namespace std;


void printUsage(char* argv0){
    cout<<"Statistics of multiple images"<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" fullPathDir1  fullPathDir2 fullPathDir3 ..."<<endl;
}

int main(int argc, char *argv[]) {

    if (argc < 2) {
        cout << "Error: at least has a pathDir as parameter." << endl;
        printUsage(argv[0]);
        return -1;
    }

    vector<string> pathVector;
    for (int i=1; i<argc; ++i){
        pathVector.push_back(argv[i]);
    }


    vector<string> imageVector;
    int numFiles = 0;
    for (vector<string>::const_iterator it= pathVector.begin(); it != pathVector.end(); ++it){
        DIR* pDir = opendir((*it).c_str());
        struct dirent* pEntry;
        while ((pEntry = readdir(pDir)) != NULL) {
            if (pEntry->d_type == DT_REG && '.' != pEntry->d_name[0]){
                imageVector.push_back((*it)+"/"+pEntry->d_name);
                ++numFiles;
            }
        }
        closedir(pDir);
    }
    cout<<"Totally read "<<numFiles <<"  image files"<<endl;

    const int Dimension = 3;
    using ImageType = itk::Image< float, Dimension >;
    using ReaderType = itk::ImageFileReader< ImageType >;
    ReaderType::Pointer reader = ReaderType::New();

    ImageType::SizeType gSize;  // g means global
    ImageType::PointType gOrigin;
    ImageType::SpacingType gSpacing;
    gSize.Fill(0);
    gOrigin.Fill(0);
    gSpacing.Fill(0);

    ImageType::SizeType minSize;
    ImageType::PointType minOrigin;
    ImageType::SpacingType minSpacing;

    ImageType::SizeType maxSize;
    ImageType::PointType maxOrigin;
    ImageType::SpacingType maxSpacing;

    ImageType::SizeType meanSize;
    ImageType::PointType meanOrigin;
    ImageType::SpacingType meanSpacing;

    meanSize.Fill(0);
    meanOrigin.Fill(0);
    meanSpacing.Fill(0);

    numFiles = 9; //debug

    int computingNumFile = 0;
    for(int i= 0; i<numFiles; ++i){
        if ("/Users/hxie1/msd/Task07_Pancreas/imagesTr/pancreas_296.nii.gz" == imageVector[i])
        {
            continue;
        }
        reader->SetFileName( imageVector[i]);
        reader->Update();
        typename ImageType::Pointer image = reader->GetOutput();

        // get Image information
        ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
        ImageType::PointType origin = image->GetOrigin();
        ImageType::SpacingType spacing = image->GetSpacing();

        gSize += size;
        for(int j=0 ; j<Dimension; ++j){
            gOrigin[j] += origin[j];
        }
        gSpacing += spacing;
        
        if (i==0){
            minSize = size;
            minOrigin = origin;
            minSpacing = spacing;

            maxSize = size;
            maxOrigin = origin;
            maxSpacing = spacing;
        }
        else{
            for(int j=0; j<Dimension; ++j){
               if (size[j] < minSize[j]) minSize[j] = size[j];
               if (origin[j] < minOrigin[j]) minOrigin[j] = origin[j];
               if (spacing[j] < minSpacing[j]) minSpacing[j] = spacing[j];

               if (size[j] > maxSize[j]) maxSize[j] = size[j];
               if (origin[j] > maxOrigin[j]) maxOrigin[j] = origin[j];
               if (spacing[j] > maxSpacing[j]) maxSpacing[j] = spacing[j];
            }
            
        }

        ++computingNumFile;

    }

    for(int j=0; j<Dimension; ++j){
        meanSize[j] = gSize[j]/computingNumFile;
        meanOrigin[j] = gOrigin[j]/computingNumFile;
        meanSpacing[j] = gSpacing[j]/computingNumFile;
    }

    cout<<"Statistics Information of Images:"<<endl;
    cout<<"Image Directories:"<<endl;
    for (int i=0; i<pathVector.size(); ++i){
        cout<<"    "<<pathVector[i]<<endl;
    }
    cout<<"Totally compute "<<computingNumFile <<"  image files"<<endl;
    cout<<"Dimension = " << Dimension <<endl;

    cout<<"Dimension     X      Y    Z"<<endl;
    cout<<"MinSize: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<minSize[j];  cout<<endl;
    cout<<"MaxSize: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<maxSize[j];  cout<<endl;
    cout<<"MeanSize: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<meanSize[j];  cout<<endl;

    cout<<"MinOrigin: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<minOrigin[j];  cout<<endl;
    cout<<"MaxOrigin: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<maxOrigin[j];  cout<<endl;
    cout<<"MeanOrigin: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<meanOrigin[j];  cout<<endl;

    cout<<"MinSpacing: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<minSpacing[j];  cout<<endl;
    cout<<"MaxSpacing: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<maxSpacing[j];  cout<<endl;
    cout<<"MeanSpacing: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<meanSpacing[j];  cout<<endl;

    cout<<"=======End of Image Files Statistics=============="<<endl;
    return 0;

}