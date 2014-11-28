#include "detect.h"

using namespace std;

inline int myRound(const float &val) {
	return val - (int)val >= 0.5 ? (int)val + 1 : (int)val;
}
inline int myMax(const int &a, const int &b) {
	return a > b ? a : b;
}
inline int myMin(const int &a, const int &b) {
	return a < b ? a : b;
}
inline int myAbs(const int &a) {
	return a < 0 ? -a : a;
}
inline bool is_equal(const myRect &r1, const myRect &r2) {

	int temp = myMin( myMin(r1.width, r1.height), myMin(r2.width, r2.height) );
	int dis = myRound(temp * GROUP_EPS);
	if( r2.x <= r1.x + dis &&
		r2.x >= r1.x - dis &&
		r2.y <= r1.y + dis &&
		r2.y >= r1.y - dis &&
		r2.width <= myRound( r1.width * (1 + GROUP_EPS) ) &&
		myRound( r2.width * (1 + GROUP_EPS) ) >= r1.width ) {
			return true;
	}
	temp = myMax( myMax(r1.width, r1.height), myMax(r2.width, r2.height) );
	dis = myRound(temp * GROUP_EPS);
	int x1 = r1.x + r1.width / 2;
	int y1 = r1.y + r1.height / 2;
	int x2 = r2.x + r2.width / 2;
	int y2 = r2.y + r2.height / 2;
	return myAbs(x1 - x2) + myAbs(y1 - y2) <= dis * 2;
}

vector<myRect> faceDetect(
	IplImage *img, 
	cascadeClassifier &cascade, 
	int minNeighbors, 
	bool judgeSkin, 
	float upscale, 
	mySize minSize, 
	mySize maxSize
) 
{
	if(maxSize.width == 0 && maxSize.height == 0) {
		maxSize = mySize(img->width, img->height);
	}
	vector<myRect> allCandidates;
	IplImage *gray_img = NULL;

	if(img->nChannels == 1) {
		gray_img = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
		cvCopy(img, gray_img, NULL);
	}
	else {
		gray_img = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
		cvCvtColor(img, gray_img, CV_BGR2GRAY);
	}
	pair<intImage, intImage> sum = buildIntImagePair(gray_img);
	intImage skin_sum;

	if(judgeSkin) {
		skin_sum = buildSkinSum(img);
	}

	for(float factor = 1.0; ; factor *= upscale) {
		int win_size = (int)(FeatureSize * factor);
		if(win_size > maxSize.height || win_size > maxSize.width) {
			break;
		}
		if(win_size < minSize.height || win_size < minSize.width) {
			continue;
		}
		setImageForCascadeClassifier(0, cascade, factor, win_size, sum.first, sum.second);
		int ystep = myMax(2, (int)factor / 2 - 1);
		int xstep = myMax(2, (int)factor);
		int height = gray_img->height - win_size;
		int width = gray_img->width - win_size;
		int square = win_size *win_size;

#pragma omp parallel for schedule (static)
		for(int y = 0; y < height; y += ystep) {
			for(int x = 0; x < width; x += xstep) {
				if(judgeSkin) {
					int x1 = x, y1 = y, x2 = x + win_size - 1, y2 = y + win_size - 1;
					//sum[y2][x2] - sum[y1 - 1][x2] - sum[y2][x1 - 1] + sum[y1 - 1][x1 - 1]
					int cnt =   skin_sum.data[y2       * skin_sum.width + x2] 
					          - skin_sum.data[(y1 - 1) * skin_sum.width + x2]
							  - skin_sum.data[y2       * skin_sum.width + x1 - 1]
							  + skin_sum.data[(y1 - 1) * skin_sum.width + x1 - 1];
					if(1.0f * cnt / square < 0.4f) {
						continue;
					}
				}
				if(judge(0, cascade, y, x, win_size, sum.first.width) == POSITIVE) {
#pragma omp critical
					{
						allCandidates.push_back(myRect(x, y, win_size, win_size));
					}
				}
			}
		}
	}
	cvReleaseImage(&gray_img);
	mergeFaces(allCandidates, minNeighbors);
	return allCandidates;
}


void mergeFaces(vector<myRect> &allCandidates, int minNeighbors) {
	if(minNeighbors == 0) return;

	vector<int> labels((int)allCandidates.size());
	int tol = partition(allCandidates, labels);
	vector<myRect> temp(tol);
	int* cnt = new int[tol];

#pragma omp parallel for schedule (static)
	for(int i = 0; i < tol; i++) {
		cnt[i] = 0;
		temp[i].x = temp[i].y = temp[i].width = temp[i].height = 0;
	}

	for(int i = 0; i < (int)allCandidates.size(); i++) {
		int idx = labels[i];
		cnt[idx]++;
		temp[idx].x += allCandidates[i].x;
		temp[idx].y += allCandidates[i].y;
		temp[idx].width += allCandidates[i].width;
		temp[idx].height += allCandidates[i].height;
	}
#pragma omp parallel for schedule (static)
	for(int i = 0; i < tol; i++) {
		if(cnt[i] >= minNeighbors) {
			temp[i].x = (temp[i].x * 2 + cnt[i]) / (2 * cnt[i]);
			temp[i].y = (temp[i].y * 2 + cnt[i]) / (2 * cnt[i]);
			temp[i].width = (temp[i].width * 2 + cnt[i]) / (2 * cnt[i]);
			temp[i].height = (temp[i].height * 2 + cnt[i]) / (2 * cnt[i]);
		}
	}
	allCandidates.swap(vector<myRect>());
#pragma omp parallel for schedule (static)
	for(int i = 0; i < tol; i++) {
		if(cnt[i] < minNeighbors) continue;
		bool flag = true;
		myRect r1 = temp[i];
		int dis = myRound(r1.width * GROUP_EPS);
		for(int j = 0; j < tol; j++) {
			if(i == j) continue;
			if(cnt[j] < minNeighbors) {
				continue;
			}
			myRect r2 = temp[j];
			if(r1.x - dis <= r2.x && r1.y - dis <= r2.y 
				&& r1.x + r1.width + dis >= r2.x + r2.width && r1.y + r1.height + dis >= r2.y + r2.height
				&& (cnt[j] >= cnt[i])) {
					flag = false;
					break;
			}
			if(r1.x >= r2.x - dis && r1.y >= r2.y - dis
				&& r1.x + r1.width <= r2.x + r2.width +dis && r1.y + r1.height <= r2.y + r2.height + dis
				&& (cnt[j] > cnt[i])) {
					flag = false;
					break;
			}
		}
		if(flag) {
#pragma omp critical
			{
				allCandidates.push_back(r1);
			}

		}
	}
	delete[] cnt;
}

intImage buildSkinSum(IplImage *img) {
	intImage ret(img->width, img->height);
	for(int i = 0; i < img->height; i++) {
		for(int j = 0; j < img->width; j++) {
			//data[i * width + j] = data[(i - 1) * width + j] + data[i * width + j - 1] - data[(i - 1) * width + j - 1] + temp;
			ret.data[i * ret.width + j]   =   ret.data[(i - 1) * ret.width + j]
			                                + ret.data[i * ret.width + j - 1]
			                                - ret.data[(i - 1) * ret.width + j - 1] 
			                                + isSkinPixel(i, j, img);
		}
	}
	return ret;
}
bool hasEnoughSkin(IplImage *img, const myRect &rect) {
	int cnt = 0;
	int x1 = rect.x, y1 = rect.y;
	int x2 = x1 + rect.width - 1, y2 = y1 + rect.height - 1;
	for(int i = y1; i <= y2; i++) {
		for(int j = x1; j <= x2; j++) {
			cnt += isSkinPixel(i, j, img);
		}
	}
	return 1.0f * cnt / (rect.width * rect.height) > 0.4f;
}
int isSkinPixel(int i, int j, IplImage *img) {
	int B = (int)cvGet2D(img, i, j).val[0];
	int G = (int)cvGet2D(img, i, j).val[1];
	int R = (int)cvGet2D(img, i, j).val[2];
	int diff = myMax(R, myMax(G, B))- myMin(R, myMin(G, B));
	return R > 95 && G > 40 && B > 20 && R > G && R > B && R - G > 15 && diff > 15;
}

int partition(std::vector<myRect>& allCandidates, std::vector<int>& labels) {
	int tol = 0;
	for(int i = 0; i < (int)allCandidates.size(); i++) {
		bool flag = false;
		for(int j = 0; j < i; j++) {
			if(is_equal(allCandidates[i], allCandidates[j])) {
				labels[i] = labels[j];
				flag = true;
				break;
			}
		}
		if(!flag) {
			labels[i] = tol++;
		}
	}
	return tol;
}
cascadeClassifier loadBinClassifier() {
	if((_access( "record_stage\\cascade_bin", 0 )) == -1) {
		errorExit("The cascade classifier file does not exist.");
	}
	cascadeClassifier cascade;
	fstream input_file;
	input_file.open("record_stage\\cascade_bin", ios::in | ios::binary); //read
	int count_stage;
	input_file.read( (char*)(&count_stage), sizeof(count_stage) );
	for(int round = 0; round < count_stage; round++) {
		cascade.classifier.push_back( vector<weakClassifier>() );
		int size;
		input_file.read( (char*)(&size), sizeof(size) );
		for(int i = 0; i < size; i++) {
			feature f;
			input_file.read( (char*)(&f.count_rect), sizeof(f.count_rect) );
			for(int j = 0; j < f.count_rect; j++) {
				input_file.read( (char*)(&f.rect[j].x), sizeof(f.rect[j].x) );
				input_file.read( (char*)(&f.rect[j].y), sizeof(f.rect[j].y) );
				input_file.read( (char*)(&f.rect[j].width), sizeof(f.rect[j].width) );
				input_file.read( (char*)(&f.rect[j].height), sizeof(f.rect[j].height) );
				input_file.read( (char*)(&f.weight[j]), sizeof(f.weight[j]) );
			}
			float threshold, left_val, right_val;
			input_file.read( (char*)(&threshold), sizeof(threshold) );
			input_file.read( (char*)(&left_val), sizeof(left_val) );
			input_file.read( (char*)(&right_val), sizeof(right_val) );
			cascade.classifier[round].push_back(weakClassifier(f, 0, threshold, 0, left_val, right_val));
		}
	}
	input_file.close();
	return cascade;
}
cascadeClassifier loadTxtClassifier() {
	if((_access( "record_stage\\cascade.txt", 0 )) == -1) {
		errorExit("The cascade classifier file does not exist.");
	}
	cascadeClassifier cascade;
	fstream input_file;
	int count_stage;
	input_file.open("record_stage\\cascade.txt");
	input_file >> count_stage;
	for(int round = 0; round < count_stage; round++) {
		cascade.classifier.push_back( vector<weakClassifier>() );
		int size;
		input_file >> size;
		for(int i = 0; i < size; i++) {
			feature f;
			input_file >> f.count_rect;
			for(int j = 0; j < f.count_rect; j++) {
				input_file >> f.rect[j].x >> f.rect[j].y >> f.rect[j].width >> f.rect[j].height >> f.weight[j];
			}
			float threshold, left_val, right_val;
			input_file >> threshold >> left_val >> right_val;
			cascade.classifier[round].push_back(weakClassifier(f, 0, threshold, 0, left_val, right_val));
		}
	}
	input_file.close();
	return cascade;
}
//====================================tracker=======================================

void faceTrackerDetect(
					   vector<myRect> &allCandidates,
					   IplImage *img, 
					   cascadeClassifier &cascade, 
					   myRect detectArea,
					   int minNeighbors, 
					   bool judgeSkin, 
					   float upscale, 
					   mySize minSize, 
					   mySize maxSize
)
{
	if(maxSize.width == 0 && maxSize.height == 0) {
		maxSize = mySize(detectArea.width, detectArea.height);
	}
	IplImage *gray_img = NULL;

	if(img->nChannels == 1) {
		gray_img = cvCreateImage(cvGetSize(img), img->depth, img->nChannels);
		cvCopy(img, gray_img, NULL);
	}
	else {
		gray_img = cvCreateImage(cvGetSize(img), IPL_DEPTH_8U, 1);
		cvCvtColor(img, gray_img, CV_BGR2GRAY);
	}
	pair<intImage, intImage> sum = buildIntImagePair(gray_img);
	intImage skin_sum;

	if(judgeSkin) {
		skin_sum = buildSkinSum(img);
	}

	for(float factor = 1.0; ; factor *= upscale) {
		int win_size = (int)(FeatureSize * factor);
		if(win_size > maxSize.height || win_size > maxSize.width) {
			break;
		}
		if(win_size < minSize.height || win_size < minSize.width) {
			continue;
		}
		setImageForCascadeClassifier(0, cascade, factor, win_size, sum.first, sum.second);
		int ystep = myMax(2, (int)factor / 2 - 1);
		int xstep = myMax(2, (int)factor / 2);

		int height = detectArea.height - win_size;
		int width = detectArea.width - win_size;
		int square = win_size * win_size;
#pragma omp parallel for schedule (static)
		for(int y = detectArea.y; y < height; y += ystep) {
			for(int x = detectArea.x; x < width; x += xstep) {
				if(judgeSkin) {
					int x1 = x, y1 = y, x2 = x + win_size - 1, y2 = y + win_size - 1;
					//sum[y2][x2] - sum[y1 - 1][x2] - sum[y2][x1 - 1] + sum[y1 - 1][x1 - 1]
					int cnt =   skin_sum.data[y2       * skin_sum.width + x2] 
					- skin_sum.data[(y1 - 1) * skin_sum.width + x2]
					- skin_sum.data[y2       * skin_sum.width + x1 - 1]
					+ skin_sum.data[(y1 - 1) * skin_sum.width + x1 - 1];
					if(1.0f * cnt / square < 0.4f) {
						continue;
					}
				}
				if(judge(0, cascade, y, x, win_size, sum.first.width) == POSITIVE) {
#pragma omp critical
					{
						allCandidates.push_back(myRect(x, y, win_size, win_size));
					}
				}
			}
		}
	}
	cvReleaseImage(&gray_img);
}

void faceTracker(
				 int cam_width, 
				 int cam_height, 
				 int cam_FPS,
				 int minNeighbors, 
				 bool judgeSkin, 
				 float upscale, 
				 mySize minSize, 
				 mySize maxSize
)
{
	IplImage* pFrame = NULL;  
	CvCapture* pCapture = cvCreateCameraCapture(-1);  
	if(pCapture == NULL) {
		puts("Can not open camera!");
		return;
	}
	cvSetCaptureProperty(pCapture, CV_CAP_PROP_FRAME_WIDTH , cam_width);
	cvSetCaptureProperty(pCapture, CV_CAP_PROP_FRAME_HEIGHT, cam_height);
	cvSetCaptureProperty(pCapture, CV_CAP_PROP_FPS, cam_FPS);

	CvScalar FaceCirclecolors = {{0, 0, 255}};

	cvNamedWindow("FaceTracker", 1); 
	cascadeClassifier cascade = loadBinClassifier();
	vector<myRect> detectArea;

	while(true) {  
		pFrame=cvQueryFrame( pCapture );  
		if(!pFrame) break;
		vector<myRect> now;
		times("");
		now = faceDetect(pFrame, cascade, minNeighbors, judgeSkin, upscale, minSize, maxSize);
		/*
		if(detectArea.size() == 0) {
			faceTrackerDetect(now, pFrame, cascade, myRect(0, 0, pFrame->width, pFrame->height), 
				minNeighbors, judgeSkin, upscale, minSize, maxSize);
		}
		else {
			for(int i = 0; i < (int)detectArea.size(); i++) {
				faceTrackerDetect(now, pFrame, cascade, detectArea[i], minNeighbors, judgeSkin, upscale, minSize, maxSize);
			}
		}
		mergeFaces(now, minNeighbors);
		*/
		times("face detection");
		cout << "                               " << now.size() << endl;
		for(int i = 0; i < (int)now.size(); i++) {
			myRect* r = &now[i];
			CvPoint center;
			int radius;
			center.x = cvRound((r->x + r->width * 0.5));
			center.y = cvRound((r->y + r->height * 0.5));
			radius = cvRound((r->width + r->height) * 0.25);
			cvCircle(pFrame, center, radius, FaceCirclecolors, 2);
		}
		cvShowImage("FaceTracker" ,pFrame);  

		/*
		detectArea.swap( vector<myRect>() );
		for(int i = 0; i < (int)now.size(); i++) {
			int x1 = now[i].x;
			int y1 = now[i].y;
			int x2 = now[i].width + x1 - 1;
			int y2 = now[i].height + y1 - 1;
			int width = (int)(now[i].width);
			int height = (int)(now[i].height);
			x1 = max(0, x1 - width);
			x2 = min(pFrame->width - 1, x2 + width);
			y1 = max(0, y1 - height);
			y2 = min(pFrame->height - 1, y2 + height);
			width = x2 - x1 + 1;
			height = y2 - y1 + 1;
			cout << now[i].x << ' ' << now[i].y << ' ' << now[i].width + now[i].x - 1 << ' ' << now[i].height + now[i].y - 1<< endl;
			cout << x1 << ' ' << y1 << ' ' << x2 << ' ' << y2 << endl;
			detectArea.push_back(myRect(x1, y1, width, height));
		}
		*/
		char c = cvWaitKey(33);  
		if(c == 27) break;  
	}  
	cvReleaseCapture(&pCapture);  
	cvDestroyWindow("FaceTracker");  
}
