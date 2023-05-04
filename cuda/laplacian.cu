#include <iostream>
#include <cmath>
#include <cstring>
#include <IL/il.h>

__global__ void grayscale(unsigned char* data, unsigned char* out, int height, int width){
    auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
    auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if(x >= width || y >= height) return;
    int i = (y*width)+x;
    int z = (height - y - 1) * width + x;
    out[z] = (307 * data[3 * i] + 604 * data[3 * i + 1] + 113 * data[3 * i + 2]) >> 10;
}

__global__ void edges(unsigned char* data, unsigned char* out, int height, int width, double strength){
    auto x = (blockIdx.x * blockDim.x) + threadIdx.x;
    auto y = (blockIdx.y * blockDim.y) + threadIdx.y;
    if(x >= width || y >= height) return;
    int i = (y*width)+x;

    unsigned char result = data[i];

    if(x > 0 && x < width-1 && y > 0 && y < height-1)
    {
        double coeff_mat[] = {0, 0, -1, 0, 0, 0, -1, -2, -1, 0, -1, -2, 16, -2, -1, 0, -1, -2, -1, 0, 0, 0, -1, 0, 0};
        int size = 5;
        int middle = (size-1)/2;

        int sum = 0;

        for(int s = 0; s < size*size; s++){
            int line = s / size;
            int offset = s % size;
            int r_offset = offset - middle;
            int r_line = line - middle;

            int rx = x + r_offset;
            int ry = y + r_line;

            sum += data[(ry * width) + rx] * (coeff_mat[s]*strength);
        }

        result = sum > 255 ? 255 : (sum < 0 ? 0 : sum);
    }else{
        result = 0;
    }

    out[i] = result;
} 

int main()
{

    unsigned int image;

    ilInit();

    ilGenImages(1, &image);
    ilBindImage(image);
    ilLoadImage("in.jpg");

    int width, height, bpp, format;

    width = ilGetInteger(IL_IMAGE_WIDTH);
    height = ilGetInteger(IL_IMAGE_HEIGHT);
    bpp = ilGetInteger(IL_IMAGE_BYTES_PER_PIXEL);
    format = ilGetInteger(IL_IMAGE_FORMAT);

    // Récupération des données de l'image
    unsigned char *data = ilGetData();

    // Traitement de l'image
    unsigned char *out_grey = new unsigned char[width * height];
    unsigned char *out_blur = new unsigned char[width * height];

    // CUDA
    unsigned char* c_data;
    unsigned char* c_out;

    // Gestion de la mesure du temps
    cudaEvent_t start, stop;
    float elapsedTime;

    // Gestion des erreurs CUDA
    cudaError_t cudaStatus;
    cudaError_t kernelStatus;

    // Creation des events pour mesure le temps
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaStatus = cudaMalloc(&c_data, 3 * width * height);
    if(cudaStatus != cudaSuccess){
        std::cout << "Erreur cudaMalloc c_data" << std::endl;
    }

    cudaStatus = cudaMalloc(&c_out, width * height);
    if(cudaStatus != cudaSuccess){
        std::cout << "Erreur cudaMalloc c_out" << std::endl;
    }

    cudaStatus = cudaMemcpy(c_data, data, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if(cudaStatus != cudaSuccess){
        std::cout << "Erreur cudaMemcpy c_data - HostToDevice" << std::endl;
    }

    dim3 blockDimension (32, 32);
    dim3 gridDimensions ((width/blockDimension.x)+1, (height/blockDimension.y)+1);

    // Debut du record de l'event start
    cudaEventRecord(start,0);

    grayscale<<<gridDimensions, blockDimension, blockDimension.x * blockDimension.y>>>(c_data, c_out, height, width);
    kernelStatus = cudaGetLastError();
    if(kernelStatus != cudaSuccess){
        std::cout << "Erreur CUDA " << cudaGetErrorString(kernelStatus) << std::endl;
    }
    
    cudaStatus = cudaMemcpy(c_data, c_out, width * height, cudaMemcpyDeviceToDevice);
    if(cudaStatus != cudaSuccess){
        std::cout << "Erreur cudaMemcpy c_data - DeviceToHost" << std::endl;
    }
    
    edges<<<gridDimensions, blockDimension, blockDimension.x * blockDimension.y>>>(c_data, c_out, height, width, 1.2f);
    kernelStatus = cudaGetLastError();
    if(kernelStatus != cudaSuccess){
        std::cout << "Erreur CUDA " << cudaGetErrorString(kernelStatus) << std::endl;
    }

    cudaStatus = cudaMemcpy(out_blur, c_out, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);
    if(cudaStatus != cudaSuccess){
        std::cout << "Erreur cudaMemcpy out_blur - DeviceToHost" << std::endl;
    }

    // Récupération du temps d'éxécution
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time (ms) : " << elapsedTime << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(c_data);
    cudaFree(c_out);

    // Placement des données dans l'image
    ilTexImage(width, height, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_blur);

    ilEnable(IL_FILE_OVERWRITE);

    ilSaveImage("out.jpg");

    ilDeleteImages(1, &image);

    delete[] out_blur;
    delete[] out_grey;
}