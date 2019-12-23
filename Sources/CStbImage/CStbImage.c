#include "CStbImage.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

unsigned char* load_image(const char* path, int* width, int* height, int* channels, int desired_channels){
    stbi_uc* pixels = stbi_load(path, width, height, channels, desired_channels);
    return pixels;
}

void free_image(void* pixels){
    stbi_image_free(pixels);
}
