#ifndef ALIGN_H
#define ALIGN_H

#include "Common.h"


// This part is moved from rpg_SVO

namespace ORB_SLAM2 {

/**
 * @brief align a pixel with reference image patch
 * @param[in] cur_img The current image
 * @param[in] ref_patch_with_boarder the patch with boarder, used to compute the gradient (or FEJ)
 * @param[in] ref_patch the patch in reference frame, by default is 64x64
 * @param[in] n_iter maximum iterations
 * @param[out] cur_px_estimate the estimated position in current image, must have an initial value
 * @return True if successful
 */
bool Align2D(
        const cv::Mat &cur_img,
        uint8_t *ref_patch_with_border,
        uint8_t *ref_patch,
        const int n_iter,
        Vector2f &cur_px_estimate,
        bool no_simd = false);

}

#endif