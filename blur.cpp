#include <iostream>
#include <cmath>
#include <cstring>
#include <IL/il.h>


int main() {

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
  unsigned char* data = ilGetData();

  // Traitement de l'image
  unsigned char* out_grey = new unsigned char[ width*height ];
  unsigned char* out_blur = new unsigned char[ width*height];

  // Grayscale
  for( std::size_t i = 0 ; i < width*height ; ++i )
  {
    // GREY = ( 307 * R + 604 * G + 113 * B ) / 1024
    out_grey[ i ] = ( 307 * data[ 3*i ]
		       + 604 * data[ 3*i+1 ]
		       + 113 * data[ 3*i+2 ]
		       ) >> 10;
  }

  // Inversion verticale de l'image
  // L'image etant inversee avec le code d'avant, on la remets dans le bon ordre
  for (int j = 0; j < height / 2; ++j) {
    for (int i = 0; i < width; ++i) {
      char save = out_grey[j * width + i];
      out_grey[j * width + i] = out_grey[(height - j - 1) * width + i];
      out_grey[(height - j - 1) * width + i] = save; 
    }
  }

    // Définition de la matrice de convolution
    double coeff_mat[] = {0, 0, 0, 5, 0, 0, 0,
                       0, 5,18,32,18, 5, 0,
                       0,18,64,100,64,18,0,
                       5,32,100,100,100,32,5,
                       0,18,64,100,64,18,0,
                       0, 5,18,32,18, 5, 0,
                       0, 0, 0, 5, 0, 0, 0};

    // Définition des dimensions de la matrice de convolution
    int size = 7;

    // Normalisation de la matrice de convolution
    int norm_sum = 0;
    for (int i = 0; i < size*size; i++) {
      norm_sum += coeff_mat[i];
    }

    for (int i = 0; i < size*size; i++) {
      coeff_mat[i] /= norm_sum;
    }

  int sum;
  int px, py;
    int middle = (size-1)/2;

  // On applique ensuite la matrice de convolution sur l'image
  for (int y = 0; y < height; y++) {
      for (int x = 0; x < width; x++) {
          sum = 0;
          // On applique la matrice de convolution sur le pixel courant
          for (int ky = -middle; ky < middle; ky++) {
              for (int kx = -middle; kx < middle; kx++) {
                  px = x + kx;
                  py = y + ky;
                  sum += out_grey[(py * width + px)] * coeff_mat[(middle+ky) * size + (middle+kx)];
              }
          }
          // On redonne ensuite les couleurs du pixel courant
          out_blur[(y*width + x)] = sum;
      }
  }

  //Placement des données dans l'image
  ilTexImage( width, height, 1, 1, IL_LUMINANCE, IL_UNSIGNED_BYTE, out_blur);

  ilEnable(IL_FILE_OVERWRITE);

  ilSaveImage("out.jpg");

  ilDeleteImages(1, &image); 

  delete [] out_blur;
  delete [] out_grey;
}