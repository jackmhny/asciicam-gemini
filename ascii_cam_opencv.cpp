/*
 * Color ASCII Webcam with OpenCV Face Detection for Linux
 *
 * Captures video from /dev/video0, uses OpenCV to detect faces,
 * converts frames to color, and displays them using the '▀' character
 * with a red box around detected faces.
 *
 * Compilation (after installing OpenCV):
 * g++ -o ascii_cam_opencv ascii_cam_opencv.cpp $(pkg-config --cflags --libs opencv4)
 * (If opencv4 is not found, try 'opencv')
 *
 * Dependencies:
 * - V4L2 development headers (e.g., libv4l-dev)
 * - OpenCV library (e.g., libopencv-dev)
 * - Haar Cascade XML file (e.g., haarcascade_frontalface_default.xml)
 *
 * Usage:
 * ./ascii_cam_opencv /path/to/your/haarcascade_frontalface_default.xml
 * Press Ctrl+C to exit.
 *
 * CRITICAL NOTES FOR TERMINAL SETUP:
 * 1. UTF-8 Encoding: Your terminal MUST be set to use UTF-8 encoding.
 * 2. Font Support: Your terminal font MUST include the '▀' (U+2580) character.
 * 3. 24-bit True Color: Your terminal MUST support 24-bit true color ANSI codes.
 * 4. ANSI Escape Codes: Your terminal must process ANSI escape codes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/videodev2.h>
#include <signal.h>
#include <termios.h>
#include <stdbool.h>
#include <vector> // For std::vector

// OpenCV Headers
#include <opencv2/objdetect.hpp>
#include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp> // Not strictly needed for terminal output but often useful

#define VIDEO_DEVICE "/dev/video0"
#define DEFAULT_CAPTURE_WIDTH  640
#define DEFAULT_CAPTURE_HEIGHT 480

static int TERM_OUTPUT_WIDTH  = 80;
static int TERM_OUTPUT_HEIGHT = 24;

#define BUFFER_COUNT 4

// Structure to hold rectangle coordinates (in terminal character cells)
typedef struct {
    int x;
    int y;
    int width;
    int height;
    bool active;
} FaceRectangle;

#define MAX_FACES_TO_DRAW 5 // Draw up to 5 detected faces
static FaceRectangle detected_terminal_faces[MAX_FACES_TO_DRAW];

const int SIM_FACE_BOX_THICKNESS = 1;

struct buffer {
    void   *start;
    size_t length;
};

static struct buffer *buffers;
static int fd_video = -1;
static volatile sig_atomic_t stop_flag = 0;

// OpenCV Face Detector
static cv::CascadeClassifier face_cascade;

void sigint_handler(int sig) {
    (void)sig;
    stop_flag = 1;
    fprintf(stderr, "\nStopping...\n");
}

static int xioctl(int fh, unsigned long request, void *arg) {
    int r;
    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}

static unsigned char clamp_val(int val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return (unsigned char)val;
}

static void yuv_to_rgb(unsigned char y, unsigned char u, unsigned char v,
                       unsigned char *r, unsigned char *g, unsigned char *b) {
    int c = y - 16;
    int d = u - 128;
    int e = v - 128;
    *r = clamp_val((298 * c + 409 * e + 128) >> 8);
    *g = clamp_val((298 * c - 100 * d - 208 * e + 128) >> 8);
    *b = clamp_val((298 * c + 516 * d + 128) >> 8);
}

static void get_yuv_at_pixel(const unsigned char *frame_data, int x, int y,
                             int cam_width, int cam_height, int bytes_per_line,
                             unsigned char *out_y, unsigned char *out_u, unsigned char *out_v) {
    if (x < 0) x = 0; if (x >= cam_width) x = cam_width - 1;
    if (y < 0) y = 0; if (y >= cam_height) y = cam_height - 1;
    int y_planar_offset = y * bytes_per_line;
    int uv_group_base_offset = y_planar_offset + (x / 2) * 4;
    *out_y = frame_data[y_planar_offset + x * 2];
    *out_u = frame_data[uv_group_base_offset + 1];
    *out_v = frame_data[uv_group_base_offset + 3];
}

/*
 * Real Face Detection using OpenCV
 */
int detect_faces_opencv(const unsigned char* yuyv_pixel_data, int cam_width, int cam_height,
                        int term_width, int term_height,
                        FaceRectangle* faces_output_buffer, int max_faces_to_output) {
    if (face_cascade.empty()) {
        // fprintf(stderr, "Error: Face cascade not loaded.\n"); // Already checked in main
        return 0;
    }

    // 1. Create an OpenCV Mat from the YUYV data.
    // The YUYV data is 2 bytes per pixel.
    cv::Mat yuyv_mat(cam_height, cam_width, CV_8UC2, (void*)yuyv_pixel_data);
    cv::Mat bgr_mat;
    cv::Mat gray_mat;

    // 2. Convert YUYV to BGR, then to Grayscale for detection
    try {
        cv::cvtColor(yuyv_mat, bgr_mat, cv::COLOR_YUV2BGR_YUY2); // Or COLOR_YUV2BGR_YUYV
        cv::cvtColor(bgr_mat, gray_mat, cv::COLOR_BGR2GRAY);
    } catch (const cv::Exception& e) {
        fprintf(stderr, "OpenCV conversion error: %s\n", e.what());
        return 0;
    }
    

    // 3. Improve contrast (optional, but often helps)
    cv::equalizeHist(gray_mat, gray_mat);

    // 4. Detect faces
    std::vector<cv::Rect> faces_cv;
    // Adjust parameters as needed: scaleFactor, minNeighbors, flags, minSize
    // Using relative minSize based on capture dimensions
    double min_face_size_ratio = 0.05; // e.g., face should be at least 5% of image height
    int min_face_dim = (int)(std::min(cam_width, cam_height) * min_face_size_ratio);
    if (min_face_dim < 20) min_face_dim = 20; // Absolute minimum

    face_cascade.detectMultiScale(gray_mat, faces_cv, 1.1, 4, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(min_face_dim, min_face_dim));


    // 5. Convert OpenCV Rects to our FaceRectangle format and scale to terminal coordinates
    int num_detected = 0;
    for (size_t i = 0; i < faces_cv.size() && num_detected < max_faces_to_output; ++i) {
        cv::Rect cv_rect = faces_cv[i];
        faces_output_buffer[num_detected].active = true;

        // Scale camera pixel coordinates to terminal character coordinates
        faces_output_buffer[num_detected].x = (cv_rect.x * term_width) / cam_width;
        faces_output_buffer[num_detected].y = (cv_rect.y * term_height) / cam_height;
        faces_output_buffer[num_detected].width = (cv_rect.width * term_width) / cam_width;
        faces_output_buffer[num_detected].height = (cv_rect.height * term_height) / cam_height;

        // Ensure minimum size for visibility in terminal
        if (faces_output_buffer[num_detected].width < SIM_FACE_BOX_THICKNESS * 2 + 1) {
            faces_output_buffer[num_detected].width = SIM_FACE_BOX_THICKNESS * 2 + 1;
        }
        if (faces_output_buffer[num_detected].height < SIM_FACE_BOX_THICKNESS * 2 + 1) {
            faces_output_buffer[num_detected].height = SIM_FACE_BOX_THICKNESS * 2 + 1;
        }
        
        // Clamp to terminal boundaries
        if (faces_output_buffer[num_detected].x < 0) faces_output_buffer[num_detected].x = 0;
        if (faces_output_buffer[num_detected].y < 0) faces_output_buffer[num_detected].y = 0;
        if (faces_output_buffer[num_detected].x + faces_output_buffer[num_detected].width > term_width) {
            faces_output_buffer[num_detected].width = term_width - faces_output_buffer[num_detected].x;
        }
        if (faces_output_buffer[num_detected].y + faces_output_buffer[num_detected].height > term_height) {
            faces_output_buffer[num_detected].height = term_height - faces_output_buffer[num_detected].y;
        }

        if (faces_output_buffer[num_detected].width > 0 && faces_output_buffer[num_detected].height > 0) {
             num_detected++;
        } else {
            faces_output_buffer[num_detected].active = false; // Mark as inactive if dimensions are zero/negative
        }
    }
    return num_detected;
}

static int init_device_video(int capture_width, int capture_height) {
    struct v4l2_capability cap;
    struct v4l2_format fmt;
    struct v4l2_requestbuffers req;

    fd_video = open(VIDEO_DEVICE, O_RDWR | O_NONBLOCK, 0);
    if (fd_video == -1) { perror("Cannot open video device"); return -1; }

    if (xioctl(fd_video, VIDIOC_QUERYCAP, &cap) == -1) { perror("VIDIOC_QUERYCAP"); close(fd_video); return -1; }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) { fprintf(stderr, "No video capture device\n"); close(fd_video); return -1; }
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) { fprintf(stderr, "No streaming support\n"); close(fd_video); return -1; }

    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = capture_width;
    fmt.fmt.pix.height      = capture_height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV; // Request YUYV
    fmt.fmt.pix.field       = V4L2_FIELD_ANY;

    if (xioctl(fd_video, VIDIOC_S_FMT, &fmt) == -1) { perror("VIDIOC_S_FMT"); close(fd_video); return -1; }
    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) { // Check if YUYV was actually set
        fprintf(stderr, "YUYV format not accepted by driver. Current format: %c%c%c%c\n",
                (fmt.fmt.pix.pixelformat & 0xFF), (fmt.fmt.pix.pixelformat >> 8) & 0xFF,
                (fmt.fmt.pix.pixelformat >> 16) & 0xFF, (fmt.fmt.pix.pixelformat >> 24) & 0xFF);
        close(fd_video); return -1;
    }

    memset(&req, 0, sizeof(req));
    req.count = BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd_video, VIDIOC_REQBUFS, &req) == -1) { perror("VIDIOC_REQBUFS"); close(fd_video); return -1; }
    if (req.count < 2) { fprintf(stderr, "Insufficient buffers\n"); close(fd_video); return -1; }

    buffers = (struct buffer*)calloc(req.count, sizeof(*buffers));
    if (!buffers) { perror("calloc buffers"); close(fd_video); return -1; }

    for (unsigned int n_buffers = 0; n_buffers < req.count; ++n_buffers) {
        struct v4l2_buffer buf_query;
        memset(&buf_query, 0, sizeof(buf_query));
        buf_query.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf_query.memory = V4L2_MEMORY_MMAP;
        buf_query.index  = n_buffers;
        if (xioctl(fd_video, VIDIOC_QUERYBUF, &buf_query) == -1) { perror("VIDIOC_QUERYBUF"); free(buffers); close(fd_video); return -1; }
        buffers[n_buffers].length = buf_query.length;
        buffers[n_buffers].start = mmap(NULL, buf_query.length, PROT_READ | PROT_WRITE, MAP_SHARED, fd_video, buf_query.m.offset);
        if (MAP_FAILED == buffers[n_buffers].start) { perror("mmap"); free(buffers); close(fd_video); return -1; }
    }
    return 0;
}

static int start_capturing() {
    enum v4l2_buf_type type;
    for (unsigned int i = 0; i < BUFFER_COUNT; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (xioctl(fd_video, VIDIOC_QBUF, &buf) == -1) { perror("VIDIOC_QBUF"); return -1;}
    }
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_video, VIDIOC_STREAMON, &type) == -1) { perror("VIDIOC_STREAMON"); return -1;}
    return 0;
}

// Corrected declaration to match expected arguments
static void process_frame_display(const unsigned char *frame_pixel_data, /* int frame_size_bytes - REMOVED */
                                  int cam_width, int cam_height, int bytes_per_line,
                                  int term_width, int term_height,
                                  const FaceRectangle* faces, int num_faces) {
    printf("\033[2J\033[H"); // Clear screen, cursor to home

    float step_x = (term_width > 0) ? (float)cam_width / term_width : cam_width;
    float step_y = (term_height > 0) ? (float)cam_height / term_height : cam_height;

    for (int i = 0; i < term_height; ++i) { // Terminal row
        for (int j = 0; j < term_width; ++j) { // Terminal column
            bool is_on_face_box_border = false;
            for (int k = 0; k < num_faces; ++k) {
                if (faces[k].active && faces[k].width > 0 && faces[k].height > 0) {
                    // Check if (j,i) is on the border of faces[k] rectangle
                    bool on_horizontal_border = (i >= faces[k].y && i < faces[k].y + SIM_FACE_BOX_THICKNESS && j >= faces[k].x && j < faces[k].x + faces[k].width) ||
                                                (i < faces[k].y + faces[k].height && i >= faces[k].y + faces[k].height - SIM_FACE_BOX_THICKNESS && j >= faces[k].x && j < faces[k].x + faces[k].width);
                    bool on_vertical_border   = (j >= faces[k].x && j < faces[k].x + SIM_FACE_BOX_THICKNESS && i >= faces[k].y && i < faces[k].y + faces[k].height) ||
                                                (j < faces[k].x + faces[k].width && j >= faces[k].x + faces[k].width - SIM_FACE_BOX_THICKNESS && i >= faces[k].y && i < faces[k].y + faces[k].height);
                    
                    if (on_horizontal_border || on_vertical_border) {
                        is_on_face_box_border = true;
                        break;
                    }
                }
            }

            if (is_on_face_box_border) {
                printf("\033[38;2;255;0;0m\033[48;2;255;0;0m▀\033[0m"); // Solid red block for border
            } else {
                int cam_x_sample = (int)(j * step_x + step_x / 2.0f);
                int cam_y_top_sample = (int)(i * step_y + step_y / 4.0f);
                int cam_y_bottom_sample = (int)(i * step_y + (step_y * 3.0f) / 4.0f);

                unsigned char y_top, u_top, v_top, r_top, g_top, b_top;
                get_yuv_at_pixel(frame_pixel_data, cam_x_sample, cam_y_top_sample, cam_width, cam_height, bytes_per_line, &y_top, &u_top, &v_top);
                yuv_to_rgb(y_top, u_top, v_top, &r_top, &g_top, &b_top);

                unsigned char y_bottom, u_bottom, v_bottom, r_bottom, g_bottom, b_bottom;
                get_yuv_at_pixel(frame_pixel_data, cam_x_sample, cam_y_bottom_sample, cam_width, cam_height, bytes_per_line, &y_bottom, &u_bottom, &v_bottom);
                yuv_to_rgb(y_bottom, u_bottom, v_bottom, &r_bottom, &g_bottom, &b_bottom);

                printf("\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm▀\033[0m",
                       r_top, g_top, b_top, r_bottom, g_bottom, b_bottom);
            }
        }
        printf("\n");
    }
    fflush(stdout);
}

static void main_loop(int term_width, int term_height) {
    struct v4l2_format fmt;
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd_video, VIDIOC_G_FMT, &fmt) == -1) { perror("VIDIOC_G_FMT main_loop"); return; }
    int actual_cam_width = fmt.fmt.pix.width;
    int actual_cam_height = fmt.fmt.pix.height;
    int actual_bytes_per_line = fmt.fmt.pix.bytesperline;
    if (actual_bytes_per_line == 0 && fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV) {
        actual_bytes_per_line = actual_cam_width * 2; // For YUYV, it's width * 2 bytes/pixel
    }
    if (actual_bytes_per_line == 0) { fprintf(stderr, "Could not get bytes_per_line\n"); return; }

    while (!stop_flag) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (xioctl(fd_video, VIDIOC_DQBUF, &buf) == -1) {
            if (errno == EAGAIN) { usleep(10000); continue; } // No buffer ready, try again
            perror("VIDIOC_DQBUF"); break;
        }

        // Perform face detection
        int num_faces_found = detect_faces_opencv(
            (const unsigned char*)buffers[buf.index].start,
            actual_cam_width, actual_cam_height,
            term_width, term_height,
            detected_terminal_faces, MAX_FACES_TO_DRAW
        );

        // Display the frame with detected faces
        // CORRECTED CALL: Removed buf.bytesused
        process_frame_display((const unsigned char*)buffers[buf.index].start,
                              actual_cam_width, actual_cam_height, actual_bytes_per_line,
                              term_width, term_height, detected_terminal_faces, num_faces_found);

        if (xioctl(fd_video, VIDIOC_QBUF, &buf) == -1) { perror("VIDIOC_QBUF"); break; }
        usleep(66000); // Approx 15 FPS. Adjust based on performance.
                       // Face detection is CPU intensive.
    }
}

static void stop_capturing() {
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (fd_video != -1) {
      if (xioctl(fd_video, VIDIOC_STREAMOFF, &type) == -1) perror("VIDIOC_STREAMOFF");
    }
}

static void uninit_device_video() {
    if (buffers) {
        for (unsigned int i = 0; i < BUFFER_COUNT; ++i) { 
            if (buffers[i].start && buffers[i].start != MAP_FAILED) {
                munmap(buffers[i].start, buffers[i].length);
            }
        }
        free(buffers); buffers = NULL;
    }
    if (fd_video != -1) { close(fd_video); fd_video = -1; }
}

void get_terminal_dimensions(int *width, int *height) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        *width = ws.ws_col;
        *height = ws.ws_row -1; // Use one less row to avoid issues with prompt
        if (*height <=0) *height = 1; // Ensure at least 1 row
    } else {
        perror("TIOCGWINSZ failed or returned invalid dimensions, using defaults");
        *width = 80;  // Default width
        *height = 24; // Default height
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_haarcascade_xml_file>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char* cascade_path = argv[1];

    struct sigaction sa;
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0;
    if (sigaction(SIGINT, &sa, NULL) == -1) { perror("sigaction"); return EXIT_FAILURE; }

    // Load the Haar cascade classifier
    if (!face_cascade.load(cascade_path)) {
        fprintf(stderr, "Error: Could not load face cascade from %s\n", cascade_path);
        return EXIT_FAILURE;
    }
    fprintf(stdout, "Successfully loaded face cascade from %s\n", cascade_path);


    get_terminal_dimensions(&TERM_OUTPUT_WIDTH, &TERM_OUTPUT_HEIGHT);
    printf("Detected terminal: %d cols, %d rows for video.\n", TERM_OUTPUT_WIDTH, TERM_OUTPUT_HEIGHT);
    printf("Attempting to use OpenCV for face detection.\n");
    sleep(2); // Give user time to read messages

    if (init_device_video(DEFAULT_CAPTURE_WIDTH, DEFAULT_CAPTURE_HEIGHT) == -1) {
        fprintf(stderr, "Device initialization failed.\n");
        uninit_device_video(); return EXIT_FAILURE;
    }
    if (start_capturing() == -1) {
        fprintf(stderr, "Failed to start capture.\n");
        uninit_device_video(); return EXIT_FAILURE;
    }

    printf("\033[?25l"); // Hide cursor
    
    main_loop(TERM_OUTPUT_WIDTH, TERM_OUTPUT_HEIGHT);

    printf("\033[?25h\033[0m\n"); // Show cursor, reset colors
    fprintf(stderr, "Cleaning up...\n");
    stop_capturing();
    uninit_device_video();
    fprintf(stderr, "Color ASCII Webcam with OpenCV stopped.\n");
    return EXIT_SUCCESS;
}


