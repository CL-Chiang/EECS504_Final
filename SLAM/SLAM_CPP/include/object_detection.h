#ifndef OBJECT_DETECTION
#define OBJECT_DETECTION
struct BBX
{
    BBX(int x, int y, int width, int height, int cls) {
        this->x = x;
        this->y = y;
        this->width = width;
        this->height = height;
        this->cls = cls;  
    }
    int x;
    int y;
    int width;
    int height;
    int cls;
};
#endif