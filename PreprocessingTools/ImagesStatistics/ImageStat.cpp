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
    cout<<"============= Statistics of multiple images ==========="<<endl;
    cout<<"This program analyzes the statistic min, mean and max of size, origin, spacing, physical size of all images in given directories."<<endl;
    cout<<"The min physicalSize will be the basis for further image uniformization."<<endl;
    cout<<"Usage: "<<endl;
    cout<<argv0<<" <fullPathDir1>  [fullPathDir2] [fullPathDir3] ..."<<endl;
    cout<<endl;
}

// vector<string> exceptionFiles= {"/Users/hxie1/msd/Task07_Pancreas/imagesTr/pancreas_296.nii.gz"};

vector<string> exceptionFiles={};

bool isExceptionFile(const string file, const vector<string> exceptionFiles){
    long N = exceptionFiles.size();
    for(int i =0; i< N; ++i){
       if (file == exceptionFiles[i]) return true;
    }
    return false;
}

int main(int argc, char *argv[]) {

    printUsage(argv[0]);
    if (argc < 2) {
        cout << "Error: at least has a pathDir as parameter." << endl;
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

    ImageType::SizeType gSize;  // g means global
    ImageType::PointType gOrigin;
    ImageType::SpacingType gSpacing;
    ImageType::PointType gPhysicalSize;
    gSize.Fill(0);
    gOrigin.Fill(0);
    gSpacing.Fill(0);
    gPhysicalSize.Fill(0);

    ImageType::SizeType minSize;
    ImageType::PointType minOrigin;
    ImageType::SpacingType minSpacing;
    ImageType::PointType minPhysicalSize;

    ImageType::SizeType maxSize;
    ImageType::PointType maxOrigin;
    ImageType::SpacingType maxSpacing;
    ImageType::PointType maxPhysicalSize;

    ImageType::SizeType meanSize;
    ImageType::PointType meanOrigin;
    ImageType::SpacingType meanSpacing;
    ImageType::PointType meanPhysicalSize; //mm

    meanSize.Fill(0);
    meanOrigin.Fill(0);
    meanSpacing.Fill(0);

    int computingNumFile = 0;
    for(int i= 0; i<numFiles; ++i){
        if (isExceptionFile(imageVector[i],exceptionFiles))
        {
            cout<<"Infor: omit file: "<<imageVector[i] <<endl;
            continue;
        }

        using ReaderType = itk::ImageFileReader< ImageType >;
        ReaderType::Pointer reader = ReaderType::New();  // a reader only can support to read 9 files, bad. therefore I put it inside loop.

        reader->SetFileName( imageVector[i]);
        reader->Update();
        ImageType::Pointer image = reader->GetOutput();

        // get Image information
        ImageType::SizeType size = image->GetLargestPossibleRegion().GetSize();
        ImageType::PointType origin = image->GetOrigin();
        ImageType::SpacingType spacing = image->GetSpacing();
        ImageType::PointType physicalSize;


        gSize += size;
        for(int j=0 ; j<Dimension; ++j){
            gOrigin[j] += origin[j];
            physicalSize[j] = size[j]*spacing[j];
            gPhysicalSize[j] += physicalSize[j];
        }
        gSpacing += spacing;
        
        if (i==0){
            minSize = size;
            minOrigin = origin;
            minSpacing = spacing;
            minPhysicalSize = physicalSize;

            maxSize = size;
            maxOrigin = origin;
            maxSpacing = spacing;
            maxPhysicalSize = physicalSize;
        }
        else{
            for(int j=0; j<Dimension; ++j){
               if (size[j] < minSize[j]) minSize[j] = size[j];
               if (origin[j] < minOrigin[j]) minOrigin[j] = origin[j];
               if (spacing[j] < minSpacing[j]) minSpacing[j] = spacing[j];
               if (physicalSize[j] < minPhysicalSize[j]) minPhysicalSize[j] = physicalSize[j];

               if (size[j] > maxSize[j]) maxSize[j] = size[j];
               if (origin[j] > maxOrigin[j]) maxOrigin[j] = origin[j];
               if (spacing[j] > maxSpacing[j]) maxSpacing[j] = spacing[j];
               if (physicalSize[j] > maxPhysicalSize[j]) maxPhysicalSize[j] = physicalSize[j];
            }
            
        }

        ++computingNumFile;

    }

    for(int j=0; j<Dimension; ++j){
        meanSize[j] = gSize[j]/computingNumFile;
        meanOrigin[j] = gOrigin[j]/computingNumFile;
        meanSpacing[j] = gSpacing[j]/computingNumFile;
        meanPhysicalSize[j] = gPhysicalSize[j]/computingNumFile;
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

    cout<<"MinPhysicalSize: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<minPhysicalSize[j];  cout<<endl;
    cout<<"MaxPhysicalSize: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<maxPhysicalSize[j];  cout<<endl;
    cout<<"MeanPhysicalSize: ";
    for (int j=0; j<Dimension; ++j)  cout<<"   "<<meanPhysicalSize[j];  cout<<endl;

    cout<<"=======End of Image Files Statistics=============="<<endl;
    return 0;

}