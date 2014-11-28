#include <cv.h>  
#include <highgui.h>  
#include "train.h"
#include "common.h"
#include "detect.h"


#define  TRAIN

int main() { 
#ifdef TRAIN
	train();
#else	
	cascadeClassifier cascade = loadBinClassifier();
	while(true) {
		cout << "input name of a picture : " << endl;
		string name;
		cin >> name;
		if(name == "end") {
			break;
		}
		name = "d:\\test_data\\" + name + ".jpg";
		if( (_access( name.c_str(), 0 )) == -1 ) {
			cout << "The file does not exist, please input again" << endl;
			continue;
		}
		IplImage *img = cvLoadImage(name.c_str(), CV_LOAD_IMAGE_UNCHANGED);

		//cout << img->height << ' ' << img->width << endl;
		float time_start = clock();
		vector<myRect> ans = faceDetect(img, cascade, 3, false, 1.1);
		float time_end = clock();

		printf("人脸个数: %d   识别用时: %d ms\n", ans.size(), (int)(time_end - time_start));
		CvScalar FaceCirclecolors[] = 
		{
			{{0, 0, 255}},
			{{0, 128, 255}},
			{{0, 255, 255}},
			{{0, 255, 0}},
			{{255, 128, 0}},
			{{255, 255, 0}},
			{{255, 0, 0}},
			{{255, 0, 255}}
		};
		for(int i = 0; i < (int)ans.size(); i++) {
			myRect* r = &ans[i];
			CvPoint center;
			int radius;
			center.x = cvRound((r->x + r->width * 0.5));
			center.y = cvRound((r->y + r->height * 0.5));
			radius = cvRound((r->width + r->height) * 0.25);
			cvCircle(img, center, radius, FaceCirclecolors[i % 8], 2);
		}

		const char *pstrWindowsTitle = "My Face Detection";
		cvNamedWindow(pstrWindowsTitle, CV_WINDOW_AUTOSIZE);
		cvShowImage(pstrWindowsTitle, img);

		cvWaitKey(0);

		cvDestroyWindow(pstrWindowsTitle);
		cvReleaseImage(&img);	
	}
#endif
}












/*
bool hh[80000];
int main() {
	char path[100];
	char buf[100];
	sprintf_s(path, sizeof(path) / sizeof(char), "record_fp\\train_fp_%d.txt", 10);
	fstream input_file;
	input_file.open(path);
	int neg_index, neg_i, neg_j;
	float neg_factor;
	memset(hh, false, sizeof(hh));
	int tol = 0;
	while(input_file >> neg_index >> neg_factor >> neg_i >> neg_j) {
		if(hh[neg_index] == false) {
			tol++;
			hh[neg_index] = true;
		}
		
		IplImage* img = cvLoadImage((neg_path + _itoa(neg_index, buf, 10) + ".png").c_str(), CV_LOAD_IMAGE_UNCHANGED);
		if(img == NULL) {
			printf("%s does not exist\n", (neg_path + _itoa(neg_index, buf, 10) + ".png").c_str());
			continue;
		}
		int win_size = (int) (neg_factor * FeatureSize);
		IplImage *sub_img = cvCreateImage(cvSize(win_size, win_size), IPL_DEPTH_8U, 1);
		int temp_height = neg_i + win_size;
		int temp_width = neg_j + win_size;
#pragma omp parallel for schedule (static)
		for(int x = neg_i; x < temp_height; x++) {
			for(int y = neg_j; y < temp_width; y++) {
				cvSet2D(sub_img, x - neg_i, y - neg_j, cvGet2D(img, x, y));
			}
		}
		const char *pstrWindowsTitle = "My Face Detection";
		cvNamedWindow(pstrWindowsTitle, CV_WINDOW_AUTOSIZE);
		cvShowImage(pstrWindowsTitle, sub_img);
		cvWaitKey(0);
		cvDestroyWindow(pstrWindowsTitle);
		cvReleaseImage(&img);
		cvReleaseImage(&sub_img);
	}
	//count = 1816  idx = 3834
	cout << "total = " << tol << endl;
	int cc = 1816;
	for(int i = 3834; i < TOTAL_NEG; i++) {
		if(hh[i]) {
			cout << "count = " << cc ++ << "  idx = " << i << endl;
			IplImage* img = cvLoadImage((neg_path + _itoa(i, buf, 10) + ".png").c_str(), CV_LOAD_IMAGE_UNCHANGED);
			const char *pstrWindowsTitle = "My Face Detection";
			cvNamedWindow(pstrWindowsTitle, CV_WINDOW_AUTOSIZE);
			cvShowImage(pstrWindowsTitle, img);
			cvWaitKey(0);
			cvDestroyWindow(pstrWindowsTitle);
			cvReleaseImage(&img);

		}
	}
}
*/