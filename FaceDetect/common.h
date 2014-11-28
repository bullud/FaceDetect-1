#ifndef COMMON_H
#define COMMON_H

#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <vector>
#include <time.h>
#include <cv.h>  
#include <highgui.h> 
#include <string>
#include <math.h>
#include <string.h>
#include <fstream>
#include <stdio.h>
#include <omp.h>
#include <windows.h>
#include <io.h>
#include <iomanip>


using namespace std;

const int FeatureSize                    = 20;

//const int TOTAL_POS                      = 21574;
//const int TOTAL_POS                      = 14384;
const int TOTAL_POS                      = 35958;

const int TOTAL_NEG                      = 76156;//71734;

const int USE_POS                        = 5000;
const int USE_NEG                        = 10000;
const int USE_SAM                        = USE_POS + USE_NEG;

const int MAX_STAGE                      = 20;
const int POSITIVE                       = 1;
const int NEGATIVE                       = -1;
const float DETECT_RATE                 = 0.999f;
const float FALSE_POSITIVE_RATE         = 0.2f;
const float TWEAK_UNIT                  = (float)(1e-2);
const float MIN_TWEAK                   = (float)(1e-5);
const float GROUP_EPS                   = 0.3f;
//const std::string pos_path               = "E:\\data\\face\\";
//const std::string pos_path               = "E:\\data\\face1\\";
const std::string pos_path               = "E:\\data\\face2\\";
const std::string neg_path               = "E:\\data\\nonface\\";

struct mySize {
	int width, height;
	mySize() : width(0), height(0) {}
	mySize(const int &__width, const int &__height) : 
	width(__width), height(__height) {}
};

struct myRect {
	int x, y, width, height;
	myRect() : x(0), y(0), width(0), height(0) {}
	myRect(const int &__x, const int &__y, const int &__width, const int &__height) : 
	x(__x), y(__y), width(__width), height(__height){}
};

struct feature {
	myRect rect[3];
	int weight[3];
	int count_rect;
	feature() { count_rect = 0; memset(rect, 0, sizeof(rect)); memset(weight, 0, sizeof(weight)); }
	feature(int ty, int x, int y, int width, int height) {
		switch(ty) {
			//const int fsize[5][2] = {{2,1}, {1,2}, {3,1}, {1,3}, {2,2}};
			case 0 : 
				rect[0] = myRect(x, y, width / 2, height);                  weight[0] = 1;
				rect[1] = myRect(x + width / 2, y, width / 2, height);      weight[1] = -1;
				rect[2] = myRect();                                         weight[2] = 0;
				count_rect = 2;
				break;
			case 1 : 
				rect[0] = myRect(x, y, width, height / 2);                  weight[0] = 1;
				rect[1] = myRect(x, y + height / 2, width, height / 2);     weight[1] = -1;
				rect[2] = myRect();                                         weight[2] = 0;
				count_rect = 2;
				break;
			case 2 : 
				rect[0] = myRect(x, y, width, height);                      weight[0] = 1;
				rect[1] = myRect(x + width / 3, y, width / 3, height);      weight[1] = -2;
				rect[2] = myRect();                                         weight[2] = 0;
				count_rect = 2;
				break;
			case 3 : 
				rect[0] = myRect(x, y, width, height);                      weight[0] = 1;
				rect[1] = myRect(x, y + height / 3, width, height / 3);     weight[1] = -2;
				rect[2] = myRect();                                         weight[2] = 0;
				count_rect = 2;
				break;
			case 4 : 
				rect[0] = myRect(x, y, width, height);                      weight[0] = 1;
				rect[1] = myRect(x + width / 2, y, width / 2, height / 2);  weight[1] = -2;
				rect[2] = myRect(x, y + height / 2, width / 2, height / 2); weight[2] = -2;
				count_rect = 3;
				break;
			default:
				printf("feature error!");
		}
	}

	void output() {
		for(int i = 0; i < count_rect; i++) {
			cout << "y1 = " << rect[i].y << "  x1 = " << rect[i].x << endl;
			cout << "height = " << rect[i].height << "  width = " << rect[i].width << endl;
			cout << "weight = " << weight[i] << endl;
			cout << endl;
			cout << endl;
		}
	}
};


struct intImage {
	int width, height;
	int *inter_data, *data;
	intImage() : width(0), height(0), inter_data(NULL), data(NULL) {}

	intImage(int __width, int __height) {
		width = __width + 1;
		height = __height + 1;
		inter_data = new int[height * width];
		data = inter_data + width + 1;
		memset(inter_data, 0, sizeof(int) * height * width);
	}

	intImage(const intImage &a) {
		width = a.width;
		height = a.height;
		inter_data = new int[height * width];
		data = inter_data + width + 1;
		int temp = width * height;
		for(int i = 0; i < temp; i++) {
			inter_data[i] = a.inter_data[i];
		}
	}
	int getSum(int y1, int x1, int y2, int x2) {
		//sum[y2][x2] - sum[y1 - 1][x2] - sum[y2][x1 - 1] + sum[y1 - 1][x1 - 1]
		return data[y2 * width + x2] - data[(y1 - 1) * width + x2] - data[y2 * width + x1 - 1] + data[(y1 - 1) *width + x1 - 1];
	}
	int getFeatureSum(const feature &f) {
		int rect_sum;
		rect_sum = getSum(
			f.rect[0].y, 
			f.rect[0].x, 
			f.rect[0].y + f.rect[0].height - 1, 
			f.rect[0].x + f.rect[0].width - 1
			) * f.weight[0];

		rect_sum += getSum(
			f.rect[1].y, 
			f.rect[1].x, 
			f.rect[1].y + f.rect[1].height - 1, 
			f.rect[1].x + f.rect[1].width - 1
			) * f.weight[1];

		if(f.count_rect == 3) {
			rect_sum += getSum(
				f.rect[2].y, 
				f.rect[2].x, 
				f.rect[2].y + f.rect[2].height - 1, 
				f.rect[2].x + f.rect[2].width - 1
				) * f.weight[2];
		}
		return rect_sum;
	}

	inline int* operator[] (int i) {
		return data + i * width;
	}
	intImage operator= (const intImage& a) {
		if(inter_data != NULL) delete[] inter_data;
		width = a.width;
		height = a.height;
		int temp = width * height;
		inter_data = new int[temp];
		data = inter_data + width + 1;
		for(int i = 0; i < temp; i++) {
			inter_data[i] = a.inter_data[i];
		}
		return *this;
	}
	~intImage() {
		//free(inter_data);
		delete[] inter_data;
	}
};

struct weakClassifier {

	feature f;
	float left_val, right_val;
	int *p0[3], *p1[3], *p2[3], *p3[3];

	float alpha;
	float threshold;
	int toggle;

	weakClassifier() {
		f = feature();
		alpha = threshold = left_val = right_val = 0.0;
		toggle = 0;
		p0[0] = p1[0] = p2[0] = p3[0] = NULL;
		p0[1] = p1[1] = p2[1] = p3[1] = NULL;
		p0[2] = p1[2] = p2[2] = p3[2] = NULL;
	}
	weakClassifier(const feature &__f, const float &__alpha, const float &__threshold, const int &__toggle) {
		f = __f;
		alpha = __alpha;
		threshold = __threshold;
		toggle = __toggle;
		left_val = right_val = 0.0;
		p0[0] = p1[0] = p2[0] = p3[0] = NULL;
		p0[1] = p1[1] = p2[1] = p3[1] = NULL;
		p0[2] = p1[2] = p2[2] = p3[2] = NULL;
	}
	weakClassifier(const feature &__f, const float &__alpha, const float &__threshold, const int &__toggle, const float &__left_val, const float &__right_val) {
		f = __f;
		alpha = __alpha;
		threshold = __threshold;
		toggle = __toggle;
		left_val = __left_val;
		right_val = __right_val;
		p0[0] = p1[0] = p2[0] = p3[0] = NULL;
		p0[1] = p1[1] = p2[1] = p3[1] = NULL;
		p0[2] = p1[2] = p2[2] = p3[2] = NULL;
	}
};
//typedef vector<weakClassifier> strongClassifier;

struct cascadeClassifier {
	int *p0, *p1, *p2, *p3;
	int *pq0, *pq1, *pq2, *pq3;
	vector< vector<weakClassifier> > classifier;
};

extern std::vector<feature> features;
void generateFeatures();
void times(const char *str);
void errorExit(const char *str);
pair<intImage, intImage> buildIntImagePair(IplImage *img);
void setImageForCascadeClassifier(int stage_start, cascadeClassifier &cascade, float factor, int win_size, intImage &sum, intImage &sqsum);
int judge(int stage_start, cascadeClassifier &cascade, int y, int x, int win_size, int sum_width);
float getVariance(const pair<intImage, intImage> &sum);



#endif