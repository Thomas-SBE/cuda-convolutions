#include <iostream>
#include <cmath>
#include <cstring>
#include <IL/il.h>

__global__ void grayscale(unsigned char* data, unsigned char* out, int height, int width){
    auto i = (blockIdx.x * blockDim.x) + threadIdx.x;
    out[i] = (307 * data[3 * i] + 604 * data[3 * i + 1] + 113 * data[3 * i + 2]) >> 10;

    __syncthreads();

    if(blockIdx.x > height/2){
        char s = out[i];
        int z = (height - blockIdx.x - 1) * blockDim.x + threadIdx.x;
        out[i] = out[z];
        out[z] = s;
    }
}

__global__ void edges(unsigned char* data, unsigned char* out, int height, int width, double strength){
    auto i = (blockIdx.x * blockDim.x) + threadIdx.x;

    unsigned char result = data[i];

    if(threadIdx.x > 0 && threadIdx.x < width-1 && blockIdx.x > 0 && blockIdx.x < height-1)
    {
        double coeff_mat[] = {0, 0, 0, 5, 0, 0, 0,
                       0, 5,18,32,18, 5, 0,
                       0,18,64,100,64,18,0,
                       5,32,100,100,100,32,5,
                       0,18,64,100,64,18,0,
                       0, 5,18,32,18, 5, 0,
                       0, 0, 0, 5, 0, 0, 0};
        int size = 7;
        int middle = (size-1)/2;

        int norm_sum = 0;
        for (int i = 0; i < size*size; i++) {
        norm_sum += coeff_mat[i];
        }

        for (int i = 0; i < size*size; i++) {
        coeff_mat[i] /= norm_sum;
        }

        int x = threadIdx.x;
        int y = blockIdx.x;
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
 
    // Gestion de la mesure du temps
    cudaEvent_t start, stop;
    float elapsedTime;

    // CUDA
    unsigned char* c_data;
    unsigned char* c_out;

    // Creation des events pour mesure le temps
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Debut du record de l'event start
    cudaEventRecord(start,0);

    cudaMalloc(&c_data, 3 * width * height);
    cudaMalloc(&c_out, width * height);

    cudaMemcpy(c_data, data, 3 * width * height * sizeof(unsigned char), cudaMemcpyHostToDevice);

    grayscale<<<height, width>>>(c_data, c_out, height, width);

    cudaMemcpy(c_data, c_out, width * height, cudaMemcpyDeviceToDevice);

    edges<<<height, width>>>(c_data, c_out, height, width, 1.2f);

    cudaMemcpy(out_blur, c_out, width*height*sizeof(unsigned char), cudaMemcpyDeviceToHost);

    // Récupération du temps d'éxécution
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    std::cout << "Time (ms) : " << elapsedTime << std::endl;

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