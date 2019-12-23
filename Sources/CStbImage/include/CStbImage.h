#ifndef CStbImage_h
#define CStbImage_h

#ifdef __cplusplus
extern "C" {
#endif

    unsigned char* load_image(const char* path, int* width, int* height, int* channels, int desired_channels);
    void free_image(void* pixels);
    
#ifdef __cplusplus
}
#endif

#endif
