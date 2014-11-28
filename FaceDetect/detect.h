#ifndef DETECT_H
#define DETECT_H

#include "common.h"

using namespace std;


vector<myRect> faceDetect(
						  IplImage *img, 
						  cascadeClassifier &cascade, 
						  int minNeighbors = 3, 
						  bool judgeSkin = false, 
						  float upscale = 1.1, 
						  mySize minSize = mySize(0, 0), 
						  mySize maxSize = mySize(0, 0)
						  );
cascadeClassifier loadTxtClassifier();
int partition(vector<myRect>& _vec, vector<int>& labels);
bool hasEnoughSkin(IplImage *img, const myRect &rect);
int isSkinPixel(int i, int j, IplImage *img);
intImage buildSkinSum(IplImage *img);
cascadeClassifier loadBinClassifier();
void mergeFaces(vector<myRect> &allCandidates, int minNeighbors);

void faceTracker(
				 int cam_width = 600, 
				 int cam_height = 600, 
				 int cam_FPS = 20,
				 int minNeighbors = 3, 
				 bool judgeSkin = false, 
				 float upscale = 1.1, 
				 mySize minSize = mySize(0, 0), 
				 mySize maxSize = mySize(0, 0)
				 );

void faceTrackerDetect(
					   vector<myRect> &allCandidates,
					   IplImage *img, 
					   cascadeClassifier &cascade, 
					   myRect detectArea,
					   int minNeighbors = 3, 
					   bool judgeSkin = false, 
					   float upscale = 1.1, 
					   mySize minSize = mySize(0, 0), 
					   mySize maxSize = mySize(0, 0)
					   );

#endif