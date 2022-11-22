#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>

#include "cufile.h"

#define KB(x) ((x)*1024L)
#define TESTFILE "/mnt/test"

__global__ void hello(char *str) {
	printf("Hello World!\n");
	printf("buf: %s\n", str);
}

__global__ void strrev(char *str, int *len) {
	int size = 0;
	while (str[size] != '\0') {
		size++;
	}
	for(int i=0;i<size/2;++i) {
		char t = str[i];
		str[i] = str[size-1-i];
		str[size-1-i] = t;
	}
	/*
	printf("buf: %s\n", str);
	printf("size: %d\n", size);
	*/
	*len = size;
}

__global__ void g_reverse_int(unsigned int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
//   return ((unsigned int)ch1 << 24) + ((unsigned int)ch2 << 16) +
//          ((unsigned int)ch3 << 8) + ch4;
}

unsigned int reverse_int(unsigned int i) {
  unsigned char ch1, ch2, ch3, ch4;
  ch1 = i & 255;
  ch2 = (i >> 8) & 255;
  ch3 = (i >> 16) & 255;
  ch4 = (i >> 24) & 255;
  return ((unsigned int)ch1 << 24) + ((unsigned int)ch2 << 16) +
         ((unsigned int)ch3 << 8) + ch4;
}

void test_stream(char* file_name) {
    int fd;
    int ret;
    
    fd = open(file_name, O_RDWR | O_DIRECT);                                

    if (fd != -1) {
        int *sys_len;
        int *gpu_len;
        char *system_buf;
        char *gpumem_buf;

        unsigned int magic_number = 0;
        unsigned int number_of_images = 0;
        unsigned int n_rows = 0;
        unsigned int n_cols = 0;


        read(fd, (char*)&magic_number, sizeof(magic_number));
        read(fd, (char*)&number_of_images, sizeof(number_of_images));
        read(fd, (char*)&n_rows, sizeof(n_rows));
        read(fd, (char*)&n_cols, sizeof(n_cols));
        magic_number = reverse_int(magic_number);
        number_of_images = reverse_int(number_of_images);
        n_rows = reverse_int(n_rows);
        n_cols = reverse_int(n_cols);

        std::cout << file_name << std::endl;
        std::cout << "magic number = " << magic_number << std::endl;
        std::cout << "number of images = " << number_of_images << std::endl;
        std::cout << "rows = " << n_rows << std::endl;
        std::cout << "cols = " << n_cols << std::endl;

        // system_buf = (char*)malloc(bufsize);
        // sys_len = (int*)malloc(KB(1));
		int bufsize = n_rows * n_cols * sizeof(char);
		int n_bufsize = n_rows * n_cols * sizeof(float);

        cudaMalloc(&gpumem_buf, bufsize);
        cudaMalloc(&gpu_len, KB(1));
		off_t file_offset = 0;
		off_t mem_offset = 0;
        CUfileDescr_t cf_desc; 
        CUfileHandle_t cf_handle;

        cuFileDriverOpen();

        cf_desc.handle.fd = fd;
        cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;
		cuFileHandleRegister(&cf_handle, &cf_desc);
		cuFileBufRegister((char*)gpumem_buf, bufsize, 0);



        // std::vector<unsigned char> image(n_rows * n_cols);
        // std::vector<float> normalized_image(n_rows * n_cols);

        for (int i = 0; i < number_of_images; i++) {
            // file.read((char*)&image[0], sizeof(unsigned char) * n_rows * n_cols);

			ret = cuFileRead(cf_handle, (char*)gpumem_buf, bufsize, file_offset, mem_offset);
			file_offset += bufsize;
			mem_offset += bufsize;

            // for (int i = 0; i < n_rows * n_cols; i++) {
            // normalized_image[i] = (float)image[i] / 255 - 0.5;
            // }
            // output.push_back(normalized_image);
        }

		cudaFree(gpumem_buf);
		cudaFree(gpu_len);		
		close(fd);
		cuFileDriverClose();
    }
}

void test(char * file_name) {
	int fd;
	int ret;
	int *sys_len;
	int *gpu_len;
	char *system_buf;
	char *gpumem_buf;

	int bufsize=KB(8);
	int parasize=KB(1);

	system_buf = (char*)malloc(bufsize);
	sys_len = (int*)malloc(parasize);

	cudaMalloc(&gpumem_buf, bufsize);
	cudaMalloc(&gpu_len, parasize);
	off_t file_offset = 0;
	off_t mem_offset = 0;

	CUfileDescr_t cf_desc; 
	CUfileHandle_t cf_handle;

	cuFileDriverOpen();
	fd = open(file_name, O_RDWR | O_DIRECT);

	cf_desc.handle.fd = fd;
	cf_desc.type = CU_FILE_HANDLE_TYPE_OPAQUE_FD;

	cuFileHandleRegister(&cf_handle, &cf_desc);
	cuFileBufRegister((char*)gpumem_buf, bufsize, 0);

	ret = cuFileRead(cf_handle, (char*)gpumem_buf, bufsize, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileRead failed : %d", ret); 
	}

	/*
	hello<<<1,1>>>(gpumem_buf);
	*/
	strrev<<<1,1>>>(gpumem_buf, gpu_len);

	cudaMemcpy(sys_len, gpu_len, parasize, cudaMemcpyDeviceToHost);
	printf("sys_len : %d\n", *sys_len); 
	ret = cuFileWrite(cf_handle, (char*)gpumem_buf, *sys_len, file_offset, mem_offset);
	if (ret < 0) {
		printf("cuFileWrite failed : %d", ret); 
	}

	cudaMemcpy(system_buf, gpumem_buf, bufsize, cudaMemcpyDeviceToHost);
	printf("system_buf: %s\n", system_buf);
	printf("See also %s\n", file_name);

	cuFileBufDeregister((char*)gpumem_buf);

	cudaFree(gpumem_buf);
	cudaFree(gpu_len);
	free(system_buf);
	free(sys_len);

	close(fd);
	cuFileDriverClose();
}

int main(int argc, char *argv[])
{
	test(argv[1]);
	// char * mnist_data="/home/steven/dev/DataLoaders_DALI/cuda-neural-network/build/mnist_data/train-images-idx3-ubyte";
	// test_stream(mnist_data);
}
