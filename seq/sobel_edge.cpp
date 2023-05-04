#include <iostream>
#include <cmath>
#include <chrono>
#include <cstring>
#include <IL/il.h>

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

    // Creation et lancement du chrono
    std::chrono::time_point<std::chrono::system_clock> start, end;
    start = std::chrono::system_clock::now();

    // Grayscale
    for (std::size_t i = 0; i < width * height; ++i)
    {
        // GREY = ( 307 * R + 604 * G + 113 * B ) / 1024
        out_grey[i] = (307 * data[3 * i] + 604 * data[3 * i + 1] + 113 * data[3 * i + 2]) >> 10;
    }

    // Inversion verticale de l'image
    // L'image etant inversee avec le code d'avant, on la remets dans le bon ordre
    for (int j = 0; j < height / 2; ++j)
    {
        for (int i = 0; i < width; ++i)
        {
            char save = out_grey[j * width + i];
            out_grey[j * width + i] = out_grey[(height - j - 1) * width + i];
            out_grey[(height - j - 1) * width + i] = save;
        }
    }

    // Intensité du filtre ?
    double strength = 1.0f;

    // Définition de la matrice de convolution
    double h_coeff_mat[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
    double v_coeff_mat[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};

    // Définition des dimensions de la matrice de convolution
    int size = 3;

    // Application de l'intensité !
    for(int i = 0; i < size*size; i++) {
        h_coeff_mat[i] *= strength;
        v_coeff_mat[i] *= strength;
    }

    int sum;
    int px, py;
    int middle = (size-1)/2;

    for(int y = 1; y < height-1; y++){
        for(int x = 1; x < width-1; x++){
            sum = 0;
            for(int s = 0; s < size*size; s++){
                int line = s / size;
                int offset = s % size;
                int r_offset = offset - middle;
                int r_line = line - middle;

                int rx = x + r_offset;
                int ry = y + r_line;

                sum += out_grey[(ry * width) + rx] * h_coeff_mat[s];
                sum += out_grey[(ry * width) + rx] * v_coeff_mat[s];
                
            }
            out_blur[(y * width) + x] = sum > 255 ? 255 : (sum < 0 ? 0 : sum);
        }
    }

    // Fin du chrono
    end = std::chrono::system_clock::now();

    // Recuperation du temps écoulé
    auto elapsedTime = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    std::cout << "Elapsed Time (ms) : " << (elapsedTime/1000.0f) << std::endl;


    // Placement des données dans l'image
    ilTexImage(width, height, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_blur);

    ilEnable(IL_FILE_OVERWRITE);

    ilSaveImage("out.jpg");

    ilDeleteImages(1, &image);

    delete[] out_blur;
    delete[] out_grey;
}