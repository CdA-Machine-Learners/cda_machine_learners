#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <algorithm>
#include <vector>
#include <math.h>

class HSV {
    public:
    double h; // Hue [0, 360]
    double s; // Saturation [0, 1]
    double v; // Value [0, 1]

    HSV() : h(0), s(0), v(0) {}
    HSV(double h, double s, double v) : h(h), s(s), v(v) {}

    HSV operator+=(const HSV& rhs) {
        h += rhs.h;
        s += rhs.s;
        v += rhs.v;
        return *this;
    }

    HSV operator/=(const double& rhs) {
        h /= rhs;
        s /= rhs;
        v /= rhs;
        return *this;
    }
};

typedef struct {
    double r; // Red [0, 1]
    double g; // Green [0, 1]
    double b; // Blue [0, 1]
} RGB;

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} RGBret;

HSV rgb_to_hsv(int r, int g, int b) {
    HSV hsv;

    double rd = r / 255.0;
    double gd = g / 255.0;
    double bd = b / 255.0;

    double cmax = std::max(std::max(rd, gd), bd);
    double cmin = std::min(std::min(rd, gd), bd);
    double delta = cmax - cmin;

    // Calculate hue
    if (delta == 0)
        hsv.h = 0;
    else if (cmax == rd)
        hsv.h = 60 * (fmod(((gd - bd) / delta), 6));
    else if (cmax == gd)
        hsv.h = 60 * (((bd - rd) / delta) + 2);
    else
        hsv.h = 60 * (((rd - gd) / delta) + 4);

    // Make sure hue is in [0, 360]
    if (hsv.h < 0)
        hsv.h += 360;

    // Calculate saturation
    if (cmax == 0)
        hsv.s = 0;
    else
        hsv.s = delta / cmax;

    // Calculate value
    hsv.v = cmax;

    return hsv;
}

RGBret hsv_to_rgb(double h, double s, double v) {
    RGB rgb;

    int i = (int)(h / 60) % 6;
    double f = h / 60 - i;
    double p = v * (1 - s);
    double q = v * (1 - f * s);
    double t = v * (1 - (1 - f) * s);

    switch (i) {
        case 0:
            rgb.r = v;
            rgb.g = t;
            rgb.b = p;
            break;
        case 1:
            rgb.r = q;
            rgb.g = v;
            rgb.b = p;
            break;
        case 2:
            rgb.r = p;
            rgb.g = v;
            rgb.b = t;
            break;
        case 3:
            rgb.r = p;
            rgb.g = q;
            rgb.b = v;
            break;
        case 4:
            rgb.r = t;
            rgb.g = p;
            rgb.b = v;
            break;
        default:
            rgb.r = v;
            rgb.g = p;
            rgb.b = q;
            break;
    }

#define FTI(f) std::min( std::max( (int)(f * 255.0), 0 ), 255 )

    //Hack to safely oncvert back... I'm out of practice
    RGBret ret;
    ret.r = FTI(rgb.r);
    ret.g = FTI(rgb.g);
    ret.b = FTI(rgb.b);
    return ret;
}

struct ChangePixel {
    int sx;
    int ex;
    int sy;
    int ey;
    int qx;
    int qy;
    double change;
    int important;
    int ideal;

    double dist( int width ) const {
        double width2 = width / 2;
        return sqrt( (sx - width2) * (sx - width2) + (sy - width2) * (sy - width2) );
    }
};

double smooth_v(double x, double y, double V, double x0, double x1, double y0, double y1) {
    const double pi2 = M_PI / 2;

    double theta = atan2(y, x);
    while (theta < 0) {
        theta += M_PI * 2;
    }

/*
    double theta45 = theta;
    while (theta45 >= pi2) {
        theta45 -= pi2;
    }
    */

    //double theta_side = tan(pi2 - theta45) >= M_PI / 4 ? pi2 - theta45 : theta45;
    //double L = sqrt(1 + theta_side * theta_side);
    //double T = (cos(D / L * M_PI) + 1) / 2.0;

    double D = fmin(sqrt(x * x + y * y), 1);
    double T = (cos(D * M_PI) + 1) / 2.0;

/*
    double x0 = 0;
    double x1 = 0.3;
    double y0 = 0.4;
    double y1 = 0.8;
    double V = 1;
    */

    double aa = 0;
    double bb = 0;
    double tt = 0;

    if (theta <= pi2) {
        aa = x1;
        bb = y1;
        tt = (cos(M_PI - (theta - pi2 * 0) * 2) + 1) / 2;
    } else if (theta <= 2 * pi2) {
        aa = y1;
        bb = x0;
        tt = (cos(M_PI - (theta - pi2 * 1) * 2) + 1) / 2;
    } else if (theta <= 3 * pi2) {
        aa = x0;
        bb = y0;
        tt = (cos(M_PI - (theta - pi2 * 2) * 2) + 1) / 2;
    } else if (theta <= 4 * M_PI) {
        aa = y0;
        bb = x1;
        tt = (cos(M_PI - (theta - pi2 * 3) * 2) + 1) / 2;
    }

    double E = aa * (1 - tt) + bb * tt;
    return V * T + E * (1 - T);
}


void apply_change_pixel( double* heights, unsigned char* data, int width, int qr_width, int pixel, ChangePixel& cp ) {
    double c_width2 = (cp.ex - cp.sx) / 2.0;
    double c_height2 = (cp.ey - cp.sy) / 2.0;

    //Now we have our change value, apply it
    for ( auto cy = cp.sy; cy < cp.ey; cy++ ) {
        for ( auto cx = cp.sx; cx < cp.ex; cx++ ) {
            auto ptr = &data[(cy*width+cx) * pixel];
            auto p = rgb_to_hsv( ptr[0], ptr[1], ptr[2] );
            p.v = std::min( std::max( p.v + cp.change, 0.0 ), 1.0 );
            if ( cp.important == 1 || cp.important == 2 ) {
                p.s = 0;
                //p.s /= 10.0;
                //p.s = (cp.ideal > 0)? 1: 0;
                //p.s = std::min( std::max( p.s + cp.change, 0.0 ), 1.0 );
                p.v = (cp.ideal > 0)? 1: 0;
                //p.h = 0;
                //cp.change * 180;
            }
            //Smooth
            else if ( false && cp.important == 0 ) {
                int x0 = (cp.qx > 0)? cp.qx - 1: cp.qx;
                int x1 = (cp.qx - 1 < qr_width)? cp.qx + 1: cp.qx;
                int y0 = (cp.qy > 0)? cp.qy - 1: cp.qy;
                int y1 = (cp.qy - 1 < qr_width)? cp.qy + 1: cp.qy;

                double centered_x = (cx - cp.sx - c_width2) / c_width2;
                double centered_y = (cy - cp.sy - c_height2) / c_height2;
                p.v = smooth_v( centered_x, centered_y, p.v,
                                heights[cp.qy * qr_width + x0], heights[cp.qy * qr_width + x1],
                                heights[y0 * qr_width + cp.qx], heights[y1 * qr_width + cp.qx] );
            }

            //Write it back
            auto rgb = hsv_to_rgb( p.h, p.s, p.v );
            ptr[0] = rgb.r;
            ptr[1] = rgb.g;
            ptr[2] = rgb.b;
        }
    }

}


int importance( int x, int y, int qr_width, int qr_height ) {
    //Corners
    if ( (x < 8 && y < 8) ||
         (x > qr_width - 9 && y < 8) ||
         (x < 8 && y > qr_height - 9) ) {
        return 1;
    }

    //Lowe right corner
    if ( x > qr_width - 10 &&
         x < qr_width - 4 &&
         y > qr_height - 10 &&
         y < qr_height - 4 ) {
        return 2;
    }

    //Bands
    if ( x == 6 || y == 6 ) {
        return 3;
    }

    //normal
    return 0;
}

double calculateMedian(const std::vector<double>& data) {
    std::vector<double> sortedData = data; // Create a copy of the original vector
    std::sort(sortedData.begin(), sortedData.end());

    size_t size = sortedData.size();
    if (size % 2 == 0) {
        // If the number of elements is even, calculate the average of the middle two elements
        return (sortedData[size / 2 - 1] + sortedData[size / 2]) / 2.0;
    } else {
        // If the number of elements is odd, return the middle element
        return sortedData[size / 2];
    }
}

//Load QrSquares
std::vector<ChangePixel> processQrSquares(
        double* heights,
        npy_intp* shape, npy_intp* qr_shape,
        unsigned char* data, unsigned char* data_1px,
        double err, double important_err ) {
    std::vector<ChangePixel> changes;

    //Setup my variable bounds
    const int width = shape[1];
    const int height = shape[0];
    const int pixel = shape[2];

    const int qr_width = qr_shape[1];
    const int qr_height = qr_shape[0];

    const double scale = shape[0] / (double)qr_shape[0];

    //Step through the 1px qr code, and calculating the hsv and scaling
    for ( auto y = 0; y < qr_height; y++ ) {
        for ( auto x = 0; x < qr_width; x++ ) {
            const int sy = static_cast<int>( std::round(y * scale));
            const int ey = std::min(static_cast<int>( std::round((y + 1) * scale)), height);
            const int sx = static_cast<int>( std::round(x * scale));
            const int ex = std::min(static_cast<int>( std::round((x + 1) * scale)), width);

            HSV hsv;

            std::vector<double> vs;
            int count = 0;
            for ( auto cy = sy; cy < ey; cy++ ) {
                for ( auto cx = sx; cx < ex; cx++ ) {
                    auto ptr = &data[(cy*width+cx) * pixel];
                    auto t = rgb_to_hsv( ptr[0], ptr[1], ptr[2] );
                    hsv += t;
                    vs.push_back( t.v );
                    count++;
                }
            }

            hsv /= count;
            auto v = hsv.v;

            //auto v = calculateMedian( vs );

            const auto ideal = data_1px[y*qr_width+x];
            //printf("Average V: %f   Ideal: %d\n", v, ideal );

            //Store the heights
            heights[y*qr_width+x] = v;

            //Force movement on the edges
            auto important = importance( x, y, qr_width, qr_height );
            double e = (important > 0)? important_err : err;

            double change = 0;
            if ( !important ) {
                //Change the value?
                if ( v > e && ideal == 0 ) {
                    change = e - v;
                }
                else if ( v < (1.0 - e) && ideal == 255 ) {
                    change = (1.0 - e) - v;
                }
                else {
                    continue; //No change needed
                }
            }
            else {
                change = (ideal == 0) ? 0 : 1;
            }

            //Update the HSV
            heights[y*qr_width+x] = std::min( std::max( v + change, 0.0 ), 1.0 );

            //Add to the changes
            changes.push_back( {sx, ex, sy, ey, x, y, change, important, ideal} );
        }
    }

    return changes;
}

extern "C" {
    void c_function(double err, double important_err, double err_rate, PyObject* array, PyObject* qr_1px) {
        PyArrayObject* np_array = (PyArrayObject*)array;
        PyArrayObject* np_1px = (PyArrayObject*)qr_1px;

        // Accessing array dimensions
        int ndim = PyArray_NDIM(np_array);
        //printf("Array dimensions: %d\n", ndim);
        int ndim_1px = PyArray_NDIM(np_1px);
        //printf("QR dimensions: %d\n", ndim_1px);

        // Print array information
        npy_intp* shape = PyArray_DIMS(np_array);
        /*
        printf("Array shape: ");
        for (int i = 0; i < ndim; ++i) {
            printf("%ld ", shape[i]);
        }
        printf("\n");
        */

        // Print array information
        npy_intp* qr_shape = PyArray_DIMS(np_1px);
        /*
        printf("QR shape: ");
        for (int i = 0; i < ndim_1px; ++i) {
            printf("%ld ", qr_shape[i]);
        }
        printf("\n");
        */

        //Setup my variable bounds
        const int width = shape[1];
        const int height = shape[0];
        const int pixel = shape[2];

        const int qr_width = qr_shape[1];
        const int qr_height = qr_shape[0];
        double heights[qr_width * qr_height];

        const double scale = shape[0] / (double)qr_shape[0];

        // Accessing array data crashes?
        //npy_intp size = PyArray_SIZE(np_array); // Crashes for some reason?
        //printf("Array data[%d]: ", size);

        auto data = (unsigned char*)PyArray_DATA(np_array);
        auto data_1px = (unsigned char*)PyArray_DATA(np_1px);

        //Step through the 1px qr code, and calculating the hsv and scaling
        auto changes = processQrSquares( heights, shape, qr_shape, data, data_1px, err, important_err );

        //Fix any importants
        const int total_rate = qr_width * qr_height;
        printf("Error rate of pixels: %f\n", changes.size() / (double)total_rate);
        for (int i = changes.size() - 1; i >= 0; i--) {
            if ( changes[i].important ) {
                apply_change_pixel( heights, data, width, qr_width, pixel, changes[i] );
                changes.erase( changes.begin() + i );
            }
        }

        //Sort by error
        //std::sort(changes.begin(), changes.end(), [qr_width](const ChangePixel& a, const ChangePixel& b){ return a.dist( qr_width ) < b.dist( qr_width ); } );
        std::sort(changes.begin(), changes.end(), [](const ChangePixel& a, const ChangePixel& b){ return std::abs(a.change) > std::abs(b.change); } );

        //Sort, and deal with the least errors, the ones that are the worse, we'll leave those along for "art"
        while ( err_rate < changes.size() / (double)total_rate ) {
            //printf("Err %f\n", changes.back().change);
            apply_change_pixel( heights, data, width, qr_width, pixel, changes.back() );
            changes.pop_back();
        }
    }
}

/*
int main() {
    // Initialize Python interpreter
    Py_Initialize();

    // Import NumPy module
    import_array();

    // Create a NumPy array in Python
    const npy_intp ptr[] = {3,3};
    PyObject* python_array = PyArray_SimpleNew(2, ptr, NPY_INT);
    int* data = (int*)PyArray_DATA((PyArrayObject*)python_array);
    for (int i = 0; i < 9; ++i) {
        data[i] = i + 1;
    }

    // Call C function with the NumPy array
    //c_function(python_array);

    // Clean up
    Py_DECREF(python_array);

    // Finalize Python interpreter
    Py_Finalize();

    return 0;
}
*/