
#include "common.h"

std::vector<feature> features;

void generateFeatures() {
	const int fsize[5][2] = {{2,1}, {1,2}, {3,1}, {1,3}, {2,2}};
	for(int i = 0; i < 5; i++) {
		int sizeX = fsize[i][0];
		int sizeY = fsize[i][1];
		for(int x = 0; x <= FeatureSize - sizeX; x++) {
			for(int y = 0; y <= FeatureSize - sizeY; y++) {
				for (int width = sizeX; width <= FeatureSize - x; width += sizeX) {
					for (int height = sizeY; height <= FeatureSize - y; height += sizeY) {
						features.push_back(feature(i, x, y, width, height));
					}
				}
			}
		}
	}
}

void times(const char *str) {
	static float pre;
	if(strlen(str)) {
		printf("%s:      %dms\n", str, (int)(clock() - pre));
	}
	else {
		pre = (float)clock();
	}
}

void errorExit(const char *str) {
	puts(str);
	exit(0);
}

pair<intImage, intImage> buildIntImagePair(IplImage *img) {
	if(img->nChannels != 1) {
		errorExit("buildIntImagePair: image is not gray");
	}
	intImage sum(img->width, img->height);
	intImage sqsum(img->width, img->height);
	for(int i = 0; i < img->height; i++) {
		for(int j = 0; j < img->width; j++) {

			int temp = (int)cvGet2D(img, i, j).val[0];
			//sum[i][j] = sum[i - 1][j] + sum[i][j - 1] - sum[i - 1][j - 1] + temp
			//data[i * width + j] = data[(i - 1) * width + j] + data[i * width + j - 1] - data[(i - 1) * width + j - 1] + temp;

			sum.data[i * sum.width + j]   =   sum.data[(i - 1) * sum.width + j]
			+ sum.data[i * sum.width + j - 1]
			- sum.data[(i - 1) * sum.width + j - 1] 
			+ temp;

			sqsum.data[i * sum.width + j] =   sqsum.data[(i - 1) * sqsum.width + j]
			+ sqsum.data[i * sqsum.width + j - 1]
			- sqsum.data[(i - 1) * sqsum.width + j - 1]
			+ temp * temp;

		}
	}
	return make_pair(sum, sqsum);
}

void setImageForCascadeClassifier(int stage_start, cascadeClassifier &cascade, float factor, int win_size, intImage &sum, intImage &sqsum) {
	int x1 = 0, y1 = 0;
	int x2 = x1 + win_size - 1, y2 = y1 + win_size - 1;
	//sum[y2][x2] - sum[y1 - 1][x2] - sum[y2][x1 - 1] + sum[y1 - 1][x1 - 1]
	//p0          - p1              - p2              + p3
	cascade.p0 = sum.data + y2       * sum.width + x2;
	cascade.p1 = sum.data + (y1 - 1) * sum.width + x2;
	cascade.p2 = sum.data + y2       * sum.width + x1 - 1;
	cascade.p3 = sum.data + (y1 - 1) * sum.width + x1 - 1;

	cascade.pq0 = sqsum.data + y2       * sqsum.width + x2;
	cascade.pq1 = sqsum.data + (y1 - 1) * sqsum.width + x2;
	cascade.pq2 = sqsum.data + y2       * sqsum.width + x1 - 1;
	cascade.pq3 = sqsum.data + (y1 - 1) * sqsum.width + x1 - 1;

	for(int stage_idx = stage_start; stage_idx < (int)cascade.classifier.size(); stage_idx++) {
		for(int weak_idx = 0; weak_idx < (int)cascade.classifier[stage_idx].size(); weak_idx++) {
			feature &f = cascade.classifier[stage_idx][weak_idx].f;
			for(int rect_idx = 0; rect_idx < f.count_rect; rect_idx++) {
				x1 = (int)(f.rect[rect_idx].x * factor);
				y1 = (int)(f.rect[rect_idx].y * factor);
				x2 = x1 + (int)(f.rect[rect_idx].width  * factor) - 1;
				y2 = y1 + (int)(f.rect[rect_idx].height * factor) - 1;

				//sum[y2][x2] - sum[y1 - 1][x2] - sum[y2][x1 - 1] + sum[y1 - 1][x1 - 1]
				//p0          - p1              - p2              + p3
				cascade.classifier[stage_idx][weak_idx].p0[rect_idx] = sum.data + y2       * sum.width + x2;
				cascade.classifier[stage_idx][weak_idx].p1[rect_idx] = sum.data + (y1 - 1) * sum.width + x2;
				cascade.classifier[stage_idx][weak_idx].p2[rect_idx] = sum.data + y2       * sum.width + x1 - 1;
				cascade.classifier[stage_idx][weak_idx].p3[rect_idx] = sum.data + (y1 - 1) * sum.width + x1 - 1;
			}
		}
	}
}

int judge(int stage_start, cascadeClassifier &cascade, int y, int x, int win_size, int sum_width) {
	int offset = y * sum_width + x;
	float square = 1.0f * win_size * win_size;
	float mean = (cascade.p0[offset] - cascade.p1[offset] - cascade.p2[offset] + cascade.p3[offset]) / square;
	float variance = (float)((cascade.pq0[offset] - cascade.pq1[offset] - cascade.pq2[offset] + cascade.pq3[offset]) / square - mean * mean);
	if(variance) {
		_mm_store_ss(&variance, _mm_sqrt_ss(_mm_load_ss(&variance)));
	}
	else {
		variance = 1;
	}
	float temp = variance * square;
	for(int stage_idx = stage_start; stage_idx < (int)cascade.classifier.size(); stage_idx++) {
		float stage_sum = 0.0;
		for(int weak_idx = 0; weak_idx < (int)cascade.classifier[stage_idx].size(); weak_idx++) {
			weakClassifier &weak = cascade.classifier[stage_idx][weak_idx];
			int rect_sum;
			rect_sum  = (weak.p0[0][offset] - weak.p1[0][offset] - weak.p2[0][offset] + weak.p3[0][offset]) * weak.f.weight[0];
			rect_sum += (weak.p0[1][offset] - weak.p1[1][offset] - weak.p2[1][offset] + weak.p3[1][offset]) * weak.f.weight[1];
			if(weak.f.count_rect == 3) {
				rect_sum += (weak.p0[2][offset] - weak.p1[2][offset] - weak.p2[2][offset] + weak.p3[2][offset]) * weak.f.weight[2];
			}
			stage_sum += (rect_sum < weak.threshold * temp ? weak.left_val : weak.right_val);
		}
		if(stage_sum < 0.0f) return NEGATIVE;
	}
	return POSITIVE;
}

float getVariance(const pair<intImage, intImage> &sum) {
	int height = sum.first.height - 1;
	int width = sum.first.width - 1;
	float square = 1.0f * height * width;
	float mean = sum.first.data[(height - 1) * sum.first.width + (width - 1)] / square;
	float variance = sum.second.data[(height - 1) * sum.first.width + (width - 1)] / square - mean * mean;

	//cout << sum.first.data[(height - 1) * sum.first.width + (width - 1)] << ' ' 
	//	<< sum.second.data[(height - 1) * sum.first.width + (width - 1)] << endl;
	//getchar();

	if(variance < 0.0) return 1.0;
	else return sqrt(variance);
}
