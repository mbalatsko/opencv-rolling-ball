import cv2
import numpy as np

"""
Fully Ported to Python from ImageJ's Background Subtractor.
Only works for 8-bit greyscale images currently.
Based on the concept of the rolling ball algorithm described
in Stanley Sternberg's article,
"Biomedical Image Processing", IEEE Computer, January 1983.
Imagine that the 2D grayscale image has a third (height) dimension by the image
value at every point in the image, creating a surface. A ball of given radius
is rolled over the bottom side of this surface; the hull of the volume
reachable by the ball is the background.
http://rsbweb.nih.gov/ij/developer/source/ij/plugin/filter/BackgroundSubtracter.java.html
"""


def subtract_background_rolling_ball(img, radius, light_background=True,
                                     use_paraboloid=False, do_presmooth=True):
    """
    Calculates and subtracts or creates background from image.

    Parameters
    ----------
    img : uint8 np array
        Image
    radius : int
        Radius of the rolling ball creating the background (actually a
                      paraboloid of rotation with the same curvature)
    light_background : bool
        Whether the image has a light background.
    do_presmooth : bool
        Whether the image should be smoothened (3x3 mean) before creating
                      the background. With smoothing, the background will not necessarily
                      be below the image data.
    use_paraboloid : bool
        Whether to use the "sliding paraboloid" algorithm.

    Returns
    -------
    img, background : uint8 np array
        Background subtracted image, Background
    """
    bs = BackgroundSubtract()
    return bs.rolling_ball_background(img, radius, light_background, use_paraboloid, do_presmooth)


class BackgroundSubtract:
    X_DIRECTION = 0
    Y_DIRECTION = 1
    DIAGONAL_1A = 2
    DIAGONAL_1B = 3
    DIAGONAL_2A = 4
    DIAGONAL_2B = 5

    def __init__(self):
        self.width = 0
        self.height = 0

        self.s_width = 0
        self.s_height = 0

    def rolling_ball_background(self, img, radius, light_background=True,
                                use_paraboloid=False, do_presmooth=True):
        """
        Calculates and subtracts or creates background from image.

        Parameters
        ----------
        img : uint8 np array
            Image
        radius : int
            Radius of the rolling ball creating the background (actually a
                          paraboloid of rotation with the same curvature)
        light_background : bool
            Whether the image has a light background.
        do_presmooth : bool
            Whether the image should be smoothened (3x3 mean) before creating
                          the background. With smoothing, the background will not necessarily
                          be below the image data.
        use_paraboloid : bool
            Whether to use the "sliding paraboloid" algorithm.

        Returns
        -------
        img, background : uint8 np array
        Background subtracted image, Background

        """
        self.height, self.width = img.shape
        self.s_height, self.s_width = img.shape

        _img = img.copy()
        if do_presmooth:
            _img = self._smooth(_img)

        _img = _img.reshape(self.height * self.width)

        invert = False
        if light_background:
            invert = True

        ball = None
        if not use_paraboloid:
            ball = RollingBall(radius)

        float_img = _img.astype('float64')
        if use_paraboloid:
            float_img = self._sliding_paraboloid_float_background(float_img, radius, invert)
        else:
            float_img = self._rolling_ball_float_background(float_img, invert, ball)

        background = float_img.astype('uint8').reshape((self.height, self.width))

        offset = 255.5 if invert else 0.5
        for p in range(0, self.width*self.height):
            value = (_img[p]&0xff) - float_img[p] + offset
            value = max((value, 0))
            value = min((value, 255))
            img[int(p / self.width), int(p % self.width)] = value
        return img, background

    def _smooth(self, img, window=3):
        """
        Applies a 3x3 mean filter to specified array.
        """
        kernel = np.ones((window, window), np.float64) / (window*window)
        img = cv2.filter2D(img, -1, kernel)
        return img

    def _rolling_ball_float_background(self, float_img, invert, ball):
        shrink = ball.shrink_factor > 1
        if invert:
            float_img = 255 - float_img

        small_img = self._shrink_image(float_img, ball.shrink_factor) if shrink else float_img
        self._roll_ball(ball, small_img)

        if shrink:
            float_img = self._enlarge_image(small_img, float_img, ball.shrink_factor)

        if invert:
            float_img = 255 - float_img
        return float_img

    def _roll_ball(self, ball, float_img):
        height, width = self.s_height, self.s_width
        z_ball = ball.data
        ball_width = ball.width
        radius = int(ball_width / 2)
        cache = [0] * (width * ball_width)

        for y in range(-radius, height + radius):
            next_line_to_write = (y + radius) % ball_width
            next_line_to_read = y + radius
            if next_line_to_read < height:
                src = next_line_to_read * width
                dest = next_line_to_write * width
                cache[dest:dest+width] = float_img[src:src+width]
                float_img[src:src+width] = float("-inf")

            y0 = max((0, y - radius))
            y_ball0 = y0 - y + radius
            y_end = y + radius
            if y_end >= height:
                y_end = height - 1
            for x in range(-radius, width+radius):
                z = float("inf")
                x0 = max((0, x - radius))
                x_ball0 = x0 - x + radius
                x_end = x + radius
                if x_end >= width:
                    x_end = width - 1

                y_ball = y_ball0
                for yp in range(y0, y_end + 1):
                    cache_pointer = (yp % ball_width) * width + x0
                    bp = x_ball0 + y_ball * ball_width
                    for xp in range(x0, x_end + 1):
                        z_reduced = cache[cache_pointer] - z_ball[bp]
                        if z > z_reduced:
                            z = z_reduced
                        cache_pointer += 1
                        bp += 1
                    y_ball += 1

                y_ball = y_ball0
                for yp in range(y0, y_end + 1):
                    p = x0 + yp * width
                    bp = x_ball0 + y_ball * ball_width
                    for xp in range(x0, x_end + 1):
                        z_min = z + z_ball[bp]
                        if float_img[p] < z_min:
                            float_img[p] = z_min
                        p += 1
                        bp += 1
                    y_ball += 1

    def _shrink_image(self, img, shrink_factor):
        height, width = self.height, self.width

        self.s_height, self.s_width = int(height / shrink_factor), int(width / shrink_factor)

        img_copy = img.reshape((height, width)).copy()
        small_img = np.ones((self.s_height, self.s_width), np.float64)

        for y in range(0, self.s_height):
            for x in range(0, self.s_width):
                x_mask_min = shrink_factor * x
                y_mask_min = shrink_factor * y
                min_value = img_copy[y_mask_min:y_mask_min + shrink_factor,
                            x_mask_min:x_mask_min + shrink_factor].min()
                small_img[y, x] = min_value
        return small_img.reshape(self.s_height * self.s_width)

    def _enlarge_image(self, small_img, float_img, shrink_factor):
        height, width = self.height, self.width
        s_height, s_width = self.s_height, self.s_width

        x_s_indices, x_weigths = self._make_interpolation_arrays(width, s_width, shrink_factor)
        y_s_indices, y_weights = self._make_interpolation_arrays(height, s_height, shrink_factor)
        line0 = [0.0] * width
        line1 = [0.0] * width
        for x in range(0, width):
            line1[x] = small_img[x_s_indices[x]] * x_weigths[x] + \
                       small_img[x_s_indices[x] + 1] * (1.0 - x_weigths[x])
        y_s_line0 = -1
        for y in range(0, height):
            if y_s_line0 < y_s_indices[y]:
                line0, line1 = line1, line0
                y_s_line0 += 1
                s_y_ptr = int((y_s_indices[y] + 1) * s_width)
                for x in range(0, width):
                    line1[x] = small_img[s_y_ptr + x_s_indices[x]] * x_weigths[x] + \
                               small_img[s_y_ptr + x_s_indices[x] + 1] * (1.0 - x_weigths[x])
            weight = y_weights[y]
            p = y * width
            for x in range(0, width):
                float_img[p] = line0[x] * weight + line1[x] * (1.0 - weight)

                p += 1
        return float_img

    def _make_interpolation_arrays(self, length, s_length, shrink_factor):
        s_indices = [0] * length
        weights = [0.0] * length
        for i in range(0, length):
            s_idx = int((i - shrink_factor / 2) / shrink_factor)
            if s_idx >= s_length - 1:
                s_idx = s_length - 2
            s_indices[i] = s_idx
            distance = (i + 0.5) / shrink_factor - (s_idx + 0.5)
            weights[i] = 1.0 - distance
        return s_indices, weights

    def _sliding_paraboloid_float_background(self, float_img, radius, invert):
        height, width = self.height, self.width
        cache = [0.0] * max((height, width))
        next_point = [0] * max((height, width))
        coeff2 = np.float64(0.5) / radius
        coeff2_diag = np.float64(1.0) / radius

        if invert:
            float_img = 255 - float_img

        self._correct_corners(float_img, coeff2, cache, next_point)
        self._filter1d(float_img, self.X_DIRECTION, coeff2, cache, next_point)
        self._filter1d(float_img, self.Y_DIRECTION, coeff2, cache, next_point)
        self._filter1d(float_img, self.X_DIRECTION, coeff2, cache, next_point)
        self._filter1d(float_img, self.DIAGONAL_1A, coeff2_diag, cache, next_point)
        self._filter1d(float_img, self.DIAGONAL_1B, coeff2_diag, cache, next_point)
        self._filter1d(float_img, self.DIAGONAL_2A, coeff2_diag, cache, next_point)
        self._filter1d(float_img, self.DIAGONAL_2B, coeff2_diag, cache, next_point)
        self._filter1d(float_img, self.DIAGONAL_1A, coeff2_diag, cache, next_point)
        self._filter1d(float_img, self.DIAGONAL_1B, coeff2_diag, cache, next_point)

        if invert:
            float_img = 255 - float_img

        return float_img

    def _correct_corners(self, float_img, coeff2, cache, next_point):
        height, width = self.height, self.width
        corners = [0] * 4
        corrected_edges = [0, 0]
        corrected_edges = self._line_slide_parabola(float_img, 0, 1, width, coeff2, cache, next_point, corrected_edges)
        corners[0] = corrected_edges[0]
        corners[1] = corrected_edges[1]
        corrected_edges = self._line_slide_parabola(float_img, (height - 1) * width, 1, width, coeff2, cache, next_point, corrected_edges)
        corners[2] = corrected_edges[0]
        corners[3] = corrected_edges[1]
        corrected_edges = self._line_slide_parabola(float_img, 0, width, height, coeff2, cache, next_point, corrected_edges)
        corners[0] += corrected_edges[0]
        corners[2] += corrected_edges[1]
        corrected_edges = self._line_slide_parabola(float_img, width - 1, width, height, coeff2, cache, next_point, corrected_edges)
        corners[1] += corrected_edges[0]
        corners[3] += corrected_edges[1]
        diag_length = min((width, height))
        coeff2_diag = 2 * coeff2
        corrected_edges = self._line_slide_parabola(float_img, 0, 1 + width, diag_length, coeff2_diag, cache, next_point, corrected_edges)
        corners[0] += corrected_edges[0]
        corrected_edges = self._line_slide_parabola(float_img, width - 1, -1 + width, diag_length, coeff2_diag, cache, next_point, corrected_edges)
        corners[1] += corrected_edges[0]
        corrected_edges = self._line_slide_parabola(float_img, (height - 1) * width, 1 - width, diag_length, coeff2_diag, cache, next_point, corrected_edges)
        corners[2] += corrected_edges[0]
        corrected_edges = self._line_slide_parabola(float_img, width * height - 1, -1 - width, diag_length, coeff2_diag, cache, next_point, corrected_edges)
        corners[3] += corrected_edges[0]

        float_img[0] = min((float_img[0], corners[0] / 3))
        float_img[width-1] = min((float_img[width-1], corners[1] / 3))
        float_img[(height-1)*width] = min((float_img[(height-1)*width], corners[2] / 3))
        float_img[width*height-1] = min((float_img[width*height-1], corners[3] / 3))

    def _line_slide_parabola(self, float_img, start, inc, length, coeff2, cache, next_point, corrected_edges):
        min_value = float("inf")
        last_point = 0
        first_corner, last_corner = length - 1, 0
        v_prev1, v_prev2 = 0., 0.
        curvature_test = 1.999 * coeff2

        p = start
        for i in range(length):
            v = float_img[p]
            cache[i] = v
            min_value = min((min_value, v))
            if i >= 2 and v_prev1 + v_prev1- v_prev2 - v < curvature_test:
                next_point[last_point] = i - 1
                last_point = i - 1
            v_prev2 = v_prev1
            v_prev1 = v

            p += inc

        next_point[last_point] = length - 1
        next_point[length - 1] = float("inf")

        i1 = 0
        while i1 < length - 1:
            v1 = cache[i1]
            min_slope = float("inf")
            i2 = 0
            search_to = length
            recalculate_limit_now = 0

            j = next_point[i1]
            while j < search_to:
                v2 = cache[j]
                slope = (v2 - v1) / (j - i1) + coeff2 * (j - i1)
                if slope < min_slope:
                    min_slope = slope
                    i2 = j
                    recalculate_limit_now = -3
                if recalculate_limit_now == 0:
                    b = 0.5 * min_slope / coeff2
                    max_search = i1 + int(b + np.sqrt(b*b + (v1 - min_value) / coeff2) + 1)
                    if 0 < max_search < search_to:
                        search_to = max_search

                j = next_point[j]
                recalculate_limit_now += 1

            if i1 == 0:
                first_corner = i2
            if i2 == length - 1:
                last_corner = i1
            p = start + (i1 + 1) * inc
            for j in range(i1 + 1, i2):
                float_img[p] = v1 + (j - i1) * (min_slope - (j - i1) * coeff2)

                p += inc
            i1 = i2
        if corrected_edges is not None:
            if 4 * first_corner >= length:
                first_corner = 0
            if 4 * (length - 1 - last_corner) >= length:
                last_corner = length - 1
            v1 = cache[first_corner]
            v2 = cache[last_corner]
            slope = (v2 - v1) / (last_corner - first_corner)
            value0 = v1 - slope * first_corner
            coeff6 = 0
            mid = 0.5 * (last_corner + first_corner)
            for i in range(int((length + 2) / 3), int(2 * length / 3) + 1):
                dx = (i - mid) * 2 / (last_corner - first_corner)
                poly6 = dx*dx*dx*dx*dx*dx - 1
                if cache[i] < value0 + slope*i + coeff6*poly6:
                    coeff6 = -(value0 + slope*i - cache[i]) / poly6
            dx = (first_corner - mid) * 2.0 / (last_corner - first_corner)
            corrected_edges[0] = value0 + coeff6*(dx*dx*dx*dx*dx*dx - 1.0) + coeff2*first_corner*first_corner
            dx = (last_corner-mid)*2.0/(last_corner-first_corner)
            corrected_edges[1] = value0 + (length-1)*slope + coeff6*(dx*dx*dx*dx*dx*dx - 1.0) + \
                                 coeff2*(length-1-last_corner)*(length-1-last_corner)
        return corrected_edges

    def _filter1d(self, float_img, direction, coeff2, cache, next_point):
        height, width = self.height, self.width
        start_line = 0
        n_lines = 0
        line_inc = 0
        point_inc = 0
        length = 0

        if direction == self.X_DIRECTION:
            n_lines = height
            line_inc = width
            point_inc = 1
            length = width
        elif direction == self.Y_DIRECTION:
            n_lines = width
            line_inc = 1
            point_inc = width
            length = height
        elif direction == self.DIAGONAL_1A:
            n_lines = width - 2
            line_inc = 1
            point_inc = width + 1
        elif direction == self.DIAGONAL_1B:
            start_line = 1
            n_lines = height - 2
            line_inc = width
            point_inc = width + 1
        elif direction == self.DIAGONAL_2A:
            start_line = 2
            n_lines = width
            line_inc = 1
            point_inc = width - 1
        elif direction == self.DIAGONAL_2B:
            start_line = 0
            n_lines = height - 2
            line_inc = width
            point_inc = width - 1
        for i in range(start_line, n_lines):
            start_pixel = i * line_inc
            if direction == self.DIAGONAL_2B:
                start_pixel += width - 1
            if direction == self.DIAGONAL_1A:
                length = min((height, width-i))
            elif direction == self.DIAGONAL_1B:
                length = min((width, height-i))
            elif direction == self.DIAGONAL_2A:
                length = min((height, i+1))
            elif direction == self.DIAGONAL_2B:
                length = min((width, height-i))
            self._line_slide_parabola(float_img, start_pixel, point_inc, length, coeff2, cache, next_point, None)


class RollingBall:
    """
        A rolling ball (or actually a square part thereof)
        Here it is also determined whether to shrink the image
    """
    def __init__(self, radius):

        self.data = []
        self.width = 0

        if radius <= 10:
            self.shrink_factor = 1
            arc_trim_per = 24
        elif radius <= 30:
            self.shrink_factor = 2
            arc_trim_per = 24
        elif radius <= 100:
            self.shrink_factor = 4
            arc_trim_per = 32
        else:
            self.shrink_factor = 8
            arc_trim_per = 40
        self.build(radius, arc_trim_per)

    def build(self, ball_radius, arc_trim_per):
        small_ball_radius = ball_radius / self.shrink_factor

        if small_ball_radius < 1:
            small_ball_radius = 1

        r_square = small_ball_radius * small_ball_radius
        x_trim = int(arc_trim_per * small_ball_radius / 100)
        half_width = round(small_ball_radius - x_trim)
        self.width = 2 * half_width + 1
        self.data = [0] * (self.width * self.width)

        p = 0
        for y in range(self.width):
            for x in range(self.width):
                x_val = x - half_width
                y_val = y - half_width

                temp = r_square - x_val * x_val - y_val * y_val
                self.data[p] = np.sqrt(temp) if temp > 0 else 0

                p += 1
