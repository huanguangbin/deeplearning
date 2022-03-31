#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>
#include <math.h>
#include <memory>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/imgproc/types_c.h"
#include <algorithm>
#include <assert.h>
#include<ctime>
using namespace torch::indexing;
typedef struct BlockLoc {
	size_t x;
	size_t y;
	size_t z;
}BL;
//std::shared_ptr<torch::jit::script::Module> module;

typedef struct
{
	int width;
	int height;
	int imgNum;
	int xSize, ySize, zSize;
	int mSizeX, mSizeY, mSizeZ;
	bool stop{ false };
	int progress{ 0 };
	float overlap_compose_rate;
	std::vector<BL> blockLocList;
	torch::Tensor outputTensor;
} OctSeg;

//��������п��������굽vector
void CalBlockLoc(OctSeg* os) {
	float step_rate = 1 - os->overlap_compose_rate;//����������step_rate=1Ϊ���ص�
	size_t patch_num = os->imgNum;
	size_t rows = os->height;
	size_t cols = os->width;
	size_t i_size = os->mSizeZ;//z��ߴ�
	size_t j_size = os->mSizeY;//y��ߴ�
	size_t k_size = os->mSizeX;//x��ߴ�
	size_t i_step = i_size;
	size_t j_step = (size_t)j_size* step_rate;
	size_t k_step = (size_t)k_size* step_rate;
	os->blockLocList.clear();
	BlockLoc loc;
	for (size_t i = 0; i < patch_num; i += i_step){
		if (i + i_step > patch_num) {
			i = patch_num - i_size;
		}
		for (size_t j = 0; j < rows; j += j_step){
			if (j + j_step > rows) {
				j = rows - j_size;
			}
			for (size_t k = 0; k < cols; k += k_step){
				if (k + k_step > cols) {
					k = cols - k_size;
				}
				loc.x = k;
				loc.y = j;
				loc.z = i;
				os->blockLocList.push_back(loc);
				if (k + k_size == cols) {
					break;
				}
			}
			if (j + j_size == rows) {
				break;
			}
		}
		if (i + i_size == patch_num) {
			break;
		}
	}
}

//��������ֵȡ��ͼƬ��
void GetBlocks(OctSeg* os, std::vector<float> data, BlockLoc loc, float* block) {
	size_t nums = os->mSizeZ;
	size_t rows = os->mSizeY;
	size_t cols = os->mSizeX;
	for (size_t i = loc.z; i < nums + loc.z; i++)
	{
		for (size_t j = loc.y; j < rows + loc.y; j++)
		{
			for (size_t k = loc.x; k < cols + loc.x; k++)
			{
				block[(i - loc.z) * rows * cols + (j - loc.y) * cols + k - loc.x] =
					data[j * os->ySize * os->xSize + i * os->xSize + k];
			}
		}
	}
}
//���ӻ�Ԥ��ͼ
cv::Mat Visualization(cv::Mat prediction_map, std::string LUT_file) {

	cv::cvtColor(prediction_map.clone(), prediction_map, CV_GRAY2BGR);
	cv::Mat label_colours = cv::imread(LUT_file, 1);
	cv::cvtColor(label_colours, label_colours, CV_RGB2BGR);
	cv::Mat output_image;
	LUT(prediction_map, label_colours, output_image);
	return output_image;
}

//����png��ʽͼƬ
void SaveImg(std::vector<cv::Mat> images, size_t total_num, std::string outpath) {
	for (size_t i = 0; i < total_num; i++)
	{
		cv::Mat visImage = Visualization(images[i], "pascal.png");
		cv::imwrite(outpath + "\\" + std::to_string(long long((i))) + ".png", visImage);
	}
}

//��ͼƬ��ת��ΪTensor��ʽ
torch::Tensor TransformData(float* block, long long num, long long rows, long long cols) {
	torch::Tensor blocks;

	blocks = torch::from_blob(block, { num ,rows,cols,1 }, torch::kFloat);//torch::kByte
	//std::cout << blocks.sizes() << std::endl;
	blocks = blocks.permute({ 3,0,1,2 });
	//std::cout << blocks.sizes() << std::endl;
	blocks = torch::unsqueeze(blocks, 0);
	blocks = blocks.to(torch::kFloat);
	return blocks;
}

//��Ԥ����ƴ�ӱ���ΪMat
void TensorToMat(std::vector<cv::Mat>& images, torch::Tensor tensor, BlockLoc loc, int step) {
	cv::Mat partMat(cv::Size(160, 160), CV_8U, tensor.data_ptr());
	cv::Rect roi_rect = cv::Rect(loc.x, loc.y, 160, 160);
	partMat.copyTo(images[loc.z + step](roi_rect));
}



// ��ʼ�������͵���r xSize = 1146, ySize = 800, zSize = 1024
// xSize, ����x��ߴ�
// ySize������y��ߴ�
// zSize������z��ߴ�
//mSizeX��ģ�����������x�ߴ�
//mSizeY��ģ�����������Y�ߴ�
//mSizeZ��ģ�����������z�ߴ�
void* OS_Init(int xSize, int ySize, int zSize)
{
	OctSeg* os = new OctSeg;
	os->stop = false;
	os->xSize = xSize;
	os->ySize = ySize;
	os->zSize = zSize;
	os->width = xSize;
	os->height = zSize;
	os->imgNum = ySize;
	os->mSizeZ = 4;
	os->mSizeY = 416;
	os->mSizeX = 416;
	os->progress = 0;
	os->overlap_compose_rate = 0.2;//�ص���������ֵ����[0��1)��0Ϊ���ص���ֵԽ���ص�����Խ��
	assert(os->overlap_compose_rate>=0&& os->overlap_compose_rate < 1);
	os->outputTensor = torch::zeros({ os->imgNum,os->height,os->width }, torch::kByte);
	assert(os->xSize >= os->mSizeX);
	assert(os->ySize >= os->mSizeZ);
	assert(os->zSize >= os->mSizeY);
	CalBlockLoc(os);
	return os;
}

// �ͷ�
// width, ͼ����
// height��ͼ��߶�
// imgNum��ͼ����Ŀ
int OS_Release(void* handle)
{
	if (handle)
		delete(handle);
	return 1;
}
//���������ֵ
uchar max(unsigned char a[], int n)
{
	uchar maxPixel= 0;
	for (int i = 0; i < n; ++i){
		if (a[i] > maxPixel) {
			maxPixel = a[i];
		}
	}
	return maxPixel;
}
//�������ֵ
float mean(std::vector<float> a, int n)
{
	float sum = std::accumulate(a.begin(),a.end(),0.0);
	return sum/n;
}


//�������׼��
float standardDev(std::vector<float> a, int n, float average)
{
	double sum = 0.0;
	for (int i = 0; i < n; i++) {
		sum += (a[i] - average) * (a[i] - average);
	}
	return sqrt(sum / n);
}

//�������λ��
unsigned char percentile(unsigned char a[], int xSize, int ySize, int zSize, float percent)
{
	int len = xSize * ySize * zSize;
	size_t step = 10000;//����ϵ��
	std::vector<unsigned char> a_copy;
	a_copy.reserve(len / step);

	for (size_t i = 0; i < len; i += step) {
		a_copy.emplace_back(a[i]);
	}
	std::sort(a_copy.begin(), a_copy.end());
	int index = std::floor(static_cast<float>(a_copy.size()) * percent - 1.0);
	if (index >= a_copy.size()) index = a_copy.size() - 1;
	if (index < 0) index = 0;
	std::cout << "percentile:" << static_cast<int>(a_copy[index]) << std::endl;
	return a_copy[index];
}

//��������z_score��һ��
std::vector<float> Norm(unsigned char* volIn, int xSize, int ySize, int zSize)
{
	int len = xSize * ySize * zSize;
	uchar maxPixel = max(volIn, len);
	std::vector<float> norm_array(len, 0);
	for (size_t i = 0; i < len; ++i) {
		norm_array[i] = static_cast<float>(volIn[i]/ static_cast<float> (maxPixel));
	}
	float average = mean(norm_array, len);
	std::cout << "average:" << average << std::endl;
	float std = standardDev(norm_array, len, average);
	std::cout << "std:" << std << std::endl;
	for (size_t i = 0; i < len; i++)
	{
		norm_array[i] = (norm_array[i] - average) / std;
	}
	return norm_array;

}
std::vector<float> Norm_1(unsigned char* volIn, int xSize, int ySize, int zSize)
{
	int len = xSize * ySize * zSize;
	float mean = 0.3383;
	float std = 0.1146;
	uchar max = 0;
	std::vector<float> a_copy(len, 0);
	for (size_t i = 0; i < len; ++i) {
		if (volIn[i] > max) {
			max = volIn[i];
		}
	}
	for (size_t i = 0; i < len; ++i) {
		a_copy[i] = (volIn[i] / float(max) - mean) / std;
	}
	return a_copy;

}

// ����OCTͼ��ָ�
int OS_OctSeg(void* handle, unsigned char* volIn, unsigned char* volOut, void(*getPro)(int) = nullptr)
{
	OctSeg* os = (OctSeg*)handle;
	using torch::jit::script::Module;
	Module module = torch::jit::load("model_tiny.pt"/*"v_norm18.pt"*/);
	std::vector<float> vol = Norm(volIn, os->xSize, os->ySize, os->zSize);
	//std::vector<float> vol = Norm_1(volIn, os->xSize, os->ySize, os->zSize);
	torch::Tensor volData = torch::tensor(vol);
	volData = volData.reshape({ os->ySize,os->xSize,os->zSize });
	volData = volData.permute({ 0,2,1 });//y,z,x
	volData = volData.unsqueeze(1);
	volData = volData.to(torch::kFloat32);
	torch::NoGradGuard no_grad;
	std::cout << "Segmenting " << os->blockLocList.size() << " batch(es)" << std::endl;
	clock_t segStart = clock();
	for (int i = 0; i < os->blockLocList.size(); i++){
		if (os->stop) {
			return -1;
		}
		int zStart = os->blockLocList[i].z;
		int yStart = os->blockLocList[i].y;
		int xStart = os->blockLocList[i].x;
		torch::Tensor input = volData.index({ Slice(zStart,zStart + os->mSizeZ),Slice(0),Slice(yStart,yStart + os->mSizeY) ,Slice(xStart,xStart + os->mSizeX) });
		input = input.to(at::kCUDA);
		//std::cout << input.sizes() << std::endl;
		torch::Tensor output = module.forward({ input }).toTensor();
		//torch::Tensor output = outputList[-1];
		output = output.squeeze(1);
		output = output > 0.5;
		output = output.to(torch::kU8);
		output = output.to(torch::kCPU);
		os->outputTensor.index({ Slice(zStart,zStart + os->mSizeZ),Slice(yStart,yStart + os->mSizeY) ,Slice(xStart,xStart + os->mSizeX) }) += output;//�п��ȵ��ӣ����Ѵ���0��ֵ��1
		os->progress = (((i + 1.0) / os->blockLocList.size()) * 10000);
		printf("Segmenting:%.2f%%\r", (float)os->progress / 100);
		if (getPro != nullptr) {
			getPro((((i + 1.0) / os->blockLocList.size()) * 10000));
		}
	}
	clock_t segEnd = clock();
	std::cout << "Segment time:" << (segEnd - segStart) / 1000.0 << "s" << std::endl;

	//����ָ���ͼ
	system("md output");
	printf("Start save images...\n");
	for (size_t i = 0; i < os->ySize; i++)
	{
		cv::Mat img(cv::Size(os->xSize, os->zSize), CV_8U, os->outputTensor[i].data_ptr());
		cv::Mat visImage = Visualization(img, "pascal.png");
		cv::imwrite("output\\" + std::to_string(long long((i))) + ".png", visImage);
	}
	//����ָ���ͼ

	unsigned char* data = (unsigned char*)os->outputTensor.data_ptr();
	clock_t assignStart = clock();
	for (size_t y = 0; y < os->ySize; y++)
		for (size_t x = 0; x < os->xSize; x++)
			for (size_t z = 0; z < os->zSize; z++) {
				volOut[y * os->zSize * os->xSize + x * os->zSize + z] = data[y * os->zSize * os->xSize + z * os->xSize + x] > 0 ? 1 : 0;
			}
	clock_t assignEnd = clock();
	std::cout << "Assignment time: " << assignEnd - assignStart << "ms" << std::endl;
	std::cout << "Segmentation complete !" << std::endl;
	return 0;
}

//��ֹ�ָ�
void OS_STOP(void* handle)
{
	OctSeg* os = (OctSeg*)handle;
	os->stop = true;
}

//��ȡ�ָ����
int OS_GetProgress(void* handle)
{
	OctSeg* os = (OctSeg*)handle;
	return os->progress;
}

//���cuda����
bool EnvsCheck() {
	std::cout << "CUDA:   " << torch::cuda::is_available() << std::endl;
	std::cout << "CUDNN:  " << torch::cuda::cudnn_is_available() << std::endl;
	std::cout << "GPU(s): " << torch::cuda::device_count() << std::endl;
	if (torch::cuda::is_available() && torch::cuda::cudnn_is_available()) {
		return true;
	}
	return false;
}

static void getProgress(int value)
{
	printf("Segmenting:%.2f%%\r", (float)value / 100);
}

void OutputImg(unsigned char* volOut, int xSize, int ySize, int zSize)
{
	system("md output_out");
	printf("Start save images...\n");
	unsigned char* tmpData = (unsigned char*)malloc(xSize * zSize * sizeof(unsigned char));
	for (size_t y = 0; y < ySize; y++)
	{
		for (size_t x = 0; x < xSize; x++)
		{
			for (size_t z = 0; z < zSize; z++)
			{
				tmpData[x * zSize + z] = volOut[y * xSize * zSize + x * zSize + z] * 255;
			}
		}
		cv::Mat img(cv::Size(zSize, xSize), CV_8U, tmpData);
		//cv::Mat visImage = Visualization(img, "pascal.png");
		cv::imwrite("output_out\\" + std::to_string(long long((y))) + ".png", img);
	}
	std::cout << "Segmentation complete !" << std::endl;
	free(tmpData);
}
int main()
{
	EnvsCheck();
	int xSize = 1146, ySize = 800, zSize = 1024;//int xSize = 800, ySize = 32, zSize = 715;//test.bin
	std::ifstream fin;
	fin.open("D:\\Dataset\\Intestinal\\raw\\Intestinal_B2_20210429_ori.raw", std::ios::binary);
	if (!fin) {
		std::cout << "open error!" << std::endl;
		return -1;
	}
	unsigned char* data = (unsigned char*)malloc(xSize * ySize * zSize * sizeof(unsigned char));
	unsigned char* out = (unsigned char*)malloc(xSize * ySize * zSize * sizeof(unsigned char));
	fin.read((char*)data, xSize * ySize * zSize * sizeof(unsigned char));
	//OutputImg1(data, xSize, ySize, zSize);
	void* handle = OS_Init(xSize, ySize, zSize);
	OS_OctSeg(handle, data, out, &getProgress);
	OutputImg(out, xSize, ySize, zSize);
	OS_Release(handle);
	free(data);
	free(out);
	return 1;
}