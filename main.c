/*
 * Color ASCII Webcam for Linux using V4L2 and Half-Block Character
 *
 * Captures video from /dev/video0, converts frames to color,
 * and displays them using the '▀' (UTF-8 TOP HALF BLOCK) character with
 * 24-bit foreground and background colors in the terminal.
 * Dynamically adjusts to terminal size.
 *
 * Compilation:
 * gcc -o ascii_cam_color ascii_cam_color.c
 *
 * Dependencies:
 * You might need to install V4L2 development headers.
 * On Debian/Ubuntu: sudo apt-get install libv4l-dev
 * On Fedora: sudo dnf install v4l-utils-devel
 *
 * Usage:
 * ./ascii_cam_color
 * Press Ctrl+C to exit.
 *
 * CRITICAL NOTES FOR TERMINAL SETUP:
 * 1. UTF-8 Encoding: Your terminal MUST be set to use UTF-8 encoding
 * to correctly display the '▀' character.
 * 2. Font Support: Your terminal font MUST include the '▀' (U+2580) character.
 * Most modern monospaced fonts for Linux do.
 * 3. 24-bit True Color: Your terminal MUST support 24-bit true color
 * ANSI escape codes (e.g., \033[38;2;R;G;Bm).
 * 4. ANSI Escape Codes: Your terminal must process ANSI escape codes.
 * If you see raw codes like "[0m" or "[2J", this is not enabled.
 *
 * Common terminals like GNOME Terminal, Konsole, Alacritty, Kitty,
 * and Windows Terminal (recent versions) generally support these features.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>             // For open, O_RDWR
#include <unistd.h>            // For close, read, write, usleep
#include <errno.h>             // For errno
#include <sys/ioctl.h>         // For ioctl (both V4L2 and terminal size)
#include <sys/mman.h>          // For mmap, munmap
#include <linux/videodev2.h>   // V4L2 structures and defines
#include <signal.h>            // For signal handling (Ctrl+C)
#include <termios.h>           // For struct winsize, TIOCGWINSZ

#define VIDEO_DEVICE "/dev/video0"
#define DEFAULT_CAPTURE_WIDTH  640 // Desired capture width
#define DEFAULT_CAPTURE_HEIGHT 480 // Desired capture height

// These will be determined dynamically from terminal size
static int TERM_OUTPUT_WIDTH  = 80;
static int TERM_OUTPUT_HEIGHT = 24;

#define BUFFER_COUNT 4    // Number of V4L2 buffers

struct buffer {
    void   *start;
    size_t length;
};

static struct buffer *buffers;
static int fd = -1;
static volatile sig_atomic_t stop_flag = 0; // Flag to stop main loop

// Signal handler for Ctrl+C
void sigint_handler(int sig) {
    (void)sig; // Unused parameter
    stop_flag = 1;
    fprintf(stderr, "\nStopping...\n");
}

// Wrapper for ioctl calls, retries on EINTR
static int xioctl(int fh, unsigned long request, void *arg) {
    int r;
    do {
        r = ioctl(fh, request, arg);
    } while (-1 == r && EINTR == errno);
    return r;
}

// Helper to clamp values between 0 and 255
static unsigned char clamp_val(int val) {
    if (val < 0) return 0;
    if (val > 255) return 255;
    return (unsigned char)val;
}

// Convert YUV (YCbCr) to RGB
static void yuv_to_rgb(unsigned char y, unsigned char u, unsigned char v,
                       unsigned char *r, unsigned char *g, unsigned char *b) {
    int c = y - 16;
    int d = u - 128;
    int e = v - 128;
    *r = clamp_val((298 * c + 409 * e + 128) >> 8);
    *g = clamp_val((298 * c - 100 * d - 208 * e + 128) >> 8);
    *b = clamp_val((298 * c + 516 * d + 128) >> 8);
}

// Get Y, U, V values for a specific pixel (x, y) from a YUYV frame
static void get_yuv_at_pixel(const unsigned char *frame_data, int x, int y,
                             int cam_width, int cam_height, int bytes_per_line,
                             unsigned char *out_y, unsigned char *out_u, unsigned char *out_v) {
    if (x < 0) x = 0;
    if (x >= cam_width) x = cam_width - 1;
    if (y < 0) y = 0;
    if (y >= cam_height) y = cam_height - 1;

    int y_planar_offset = y * bytes_per_line;
    // For YUYV (Y0 U0 Y1 V0): U and V are shared by two Y values.
    // Y index is x*2. UV group starts at (x/2)*4.
    int uv_group_base_offset = y_planar_offset + (x / 2) * 4;

    *out_y = frame_data[y_planar_offset + x * 2];
    *out_u = frame_data[uv_group_base_offset + 1];
    *out_v = frame_data[uv_group_base_offset + 3];
}

// Initializes the video device
static int init_device(int capture_width, int capture_height) {
    struct v4l2_capability cap;
    struct v4l2_format fmt;
    struct v4l2_requestbuffers req;

    fd = open(VIDEO_DEVICE, O_RDWR | O_NONBLOCK, 0);
    if (fd == -1) {
        perror("Cannot open video device");
        return -1;
    }

    if (xioctl(fd, VIDIOC_QUERYCAP, &cap) == -1) {
        perror("VIDIOC_QUERYCAP");
        close(fd);
        return -1;
    }
    if (!(cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
        fprintf(stderr, "%s is no video capture device\n", VIDEO_DEVICE);
        close(fd);
        return -1;
    }
    if (!(cap.capabilities & V4L2_CAP_STREAMING)) {
        fprintf(stderr, "%s does not support streaming i/o\n", VIDEO_DEVICE);
        close(fd);
        return -1;
    }

    memset(&fmt, 0, sizeof(fmt));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width       = capture_width;
    fmt.fmt.pix.height      = capture_height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    fmt.fmt.pix.field       = V4L2_FIELD_ANY;

    if (xioctl(fd, VIDIOC_S_FMT, &fmt) == -1) {
        perror("VIDIOC_S_FMT failed");
        // Try to get current format if S_FMT fails
        if (xioctl(fd, VIDIOC_G_FMT, &fmt) == 0) {
            fprintf(stderr, "VIDIOC_S_FMT failed. Current format: %c%c%c%c, %ux%u\n",
                (fmt.fmt.pix.pixelformat & 0xFF), (fmt.fmt.pix.pixelformat >> 8) & 0xFF,
                (fmt.fmt.pix.pixelformat >> 16) & 0xFF, (fmt.fmt.pix.pixelformat >> 24) & 0xFF,
                fmt.fmt.pix.width, fmt.fmt.pix.height);
        } else {
            perror("VIDIOC_G_FMT also failed");
        }
        close(fd);
        return -1;
    }

    if (fmt.fmt.pix.pixelformat != V4L2_PIX_FMT_YUYV) {
        fprintf(stderr, "Webcam did not accept YUYV format. Current format: %c%c%c%c. Cannot proceed.\n",
                (fmt.fmt.pix.pixelformat & 0xFF), (fmt.fmt.pix.pixelformat >> 8) & 0xFF,
                (fmt.fmt.pix.pixelformat >> 16) & 0xFF, (fmt.fmt.pix.pixelformat >> 24) & 0xFF);
        close(fd);
        return -1;
    }
    // Note: Actual width/height might be different from requested.
    // We will query it again in main_loop with VIDIOC_G_FMT.

    memset(&req, 0, sizeof(req));
    req.count = BUFFER_COUNT;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;

    if (xioctl(fd, VIDIOC_REQBUFS, &req) == -1) {
        perror("VIDIOC_REQBUFS");
        close(fd);
        return -1;
    }
    if (req.count < 2) {
        fprintf(stderr, "Insufficient buffer memory on %s\n", VIDEO_DEVICE);
        close(fd);
        return -1;
    }

    buffers = calloc(req.count, sizeof(*buffers));
    if (!buffers) {
        perror("Out of memory (calloc for buffers)");
        close(fd);
        return -1;
    }

    for (unsigned int n_buffers = 0; n_buffers < req.count; ++n_buffers) {
        struct v4l2_buffer buf_query; // Renamed to avoid conflict with loop variable
        memset(&buf_query, 0, sizeof(buf_query));
        buf_query.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf_query.memory = V4L2_MEMORY_MMAP;
        buf_query.index  = n_buffers;

        if (xioctl(fd, VIDIOC_QUERYBUF, &buf_query) == -1) {
            perror("VIDIOC_QUERYBUF");
            for (unsigned int i = 0; i < n_buffers; ++i) if(buffers[i].start && buffers[i].start != MAP_FAILED) munmap(buffers[i].start, buffers[i].length);
            free(buffers);
            close(fd);
            return -1;
        }

        buffers[n_buffers].length = buf_query.length;
        buffers[n_buffers].start = mmap(NULL, buf_query.length,
                                        PROT_READ | PROT_WRITE,
                                        MAP_SHARED,
                                        fd, buf_query.m.offset);

        if (MAP_FAILED == buffers[n_buffers].start) {
            perror("mmap");
            for (unsigned int i = 0; i < n_buffers; ++i) if(buffers[i].start && buffers[i].start != MAP_FAILED) munmap(buffers[i].start, buffers[i].length);
            free(buffers);
            close(fd);
            return -1;
        }
    }
    return 0; // Success
}

// Starts video capture
static int start_capturing() {
    enum v4l2_buf_type type;
    for (unsigned int i = 0; i < BUFFER_COUNT; ++i) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index  = i;
        if (xioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            perror("VIDIOC_QBUF");
            return -1;
        }
    }
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_STREAMON, &type) == -1) {
        perror("VIDIOC_STREAMON");
        return -1;
    }
    return 0; // Success
}

// Processes a single frame and prints ASCII art with colors
static void process_frame(const void *p, int size, int cam_width, int cam_height, int bytes_per_line,
                          int term_width, int term_height) {
    (void)size; // Unused
    const unsigned char *frame_data = (const unsigned char *)p;

    // Clear screen and move cursor to top-left
    printf("\033[2J\033[H");

    // Calculate step for camera pixels per terminal character cell
    // Ensure these are not zero to avoid division by zero if terminal is too small
    float step_x = (term_width > 0) ? (float)cam_width / term_width : cam_width;
    float step_y = (term_height > 0) ? (float)cam_height / term_height : cam_height;


    for (int i = 0; i < term_height; ++i) { // Terminal row
        for (int j = 0; j < term_width; ++j) { // Terminal column
            
            // Calculate corresponding camera pixel coordinates for sampling
            // Sample center of the horizontal block for this terminal character
            int cam_x_sample = (int)(j * step_x + step_x / 2.0f);

            // Sample top quarter of the vertical block for top color
            int cam_y_top_sample = (int)(i * step_y + step_y / 4.0f);
            
            // Sample bottom (3/4 point) of the vertical block for bottom color
            int cam_y_bottom_sample = (int)(i * step_y + (step_y * 3.0f) / 4.0f);


            unsigned char y_top, u_top, v_top;
            unsigned char r_top, g_top, b_top;
            get_yuv_at_pixel(frame_data, cam_x_sample, cam_y_top_sample, cam_width, cam_height, bytes_per_line, &y_top, &u_top, &v_top);
            yuv_to_rgb(y_top, u_top, v_top, &r_top, &g_top, &b_top);

            unsigned char y_bottom, u_bottom, v_bottom;
            unsigned char r_bottom, g_bottom, b_bottom;
            get_yuv_at_pixel(frame_data, cam_x_sample, cam_y_bottom_sample, cam_width, cam_height, bytes_per_line, &y_bottom, &u_bottom, &v_bottom);
            yuv_to_rgb(y_bottom, u_bottom, v_bottom, &r_bottom, &g_bottom, &b_bottom);

            // Print: Set FG color, Set BG color, Print UTF-8 '▀', Reset colors
            // The '▀' character (U+2580) should be handled by printf if source is UTF-8.
            printf("\033[38;2;%d;%d;%dm\033[48;2;%d;%d;%dm▀\033[0m",
                   r_top, g_top, b_top,
                   r_bottom, g_bottom, b_bottom);
        }
        printf("\n");
    }
    fflush(stdout);
}

// Main capture loop
static void main_loop(int term_width, int term_height) {
    struct v4l2_format fmt;
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(fd, VIDIOC_G_FMT, &fmt) == -1) {
        perror("VIDIOC_G_FMT in main_loop failed");
        return;
    }
    int actual_cam_width = fmt.fmt.pix.width;
    int actual_cam_height = fmt.fmt.pix.height;
    int actual_bytes_per_line = fmt.fmt.pix.bytesperline;

    if (actual_bytes_per_line == 0 && fmt.fmt.pix.pixelformat == V4L2_PIX_FMT_YUYV) {
        actual_bytes_per_line = actual_cam_width * 2;
    }
    if (actual_bytes_per_line == 0) {
         fprintf(stderr, "Error: Could not determine bytes_per_line for the video format.\n");
         return;
    }

    while (!stop_flag) {
        struct v4l2_buffer buf;
        memset(&buf, 0, sizeof(buf));
        buf.type   = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

        if (xioctl(fd, VIDIOC_DQBUF, &buf) == -1) {
            if (errno == EAGAIN) {
                usleep(10000); 
                continue;
            }
            perror("VIDIOC_DQBUF");
            break; 
        }

        process_frame(buffers[buf.index].start, buf.bytesused, actual_cam_width, actual_cam_height, actual_bytes_per_line, term_width, term_height);

        if (xioctl(fd, VIDIOC_QBUF, &buf) == -1) {
            perror("VIDIOC_QBUF");
            break;
        }
        usleep(33000); // Approx 30 FPS
    }
}

// Stops video capture
static void stop_capturing() {
    enum v4l2_buf_type type;
    type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (fd != -1) {
      if (xioctl(fd, VIDIOC_STREAMOFF, &type) == -1) {
          perror("VIDIOC_STREAMOFF");
      }
    }
}

// Uninitializes the device
static void uninit_device() {
    if (buffers) {
        // req.count isn't available here, use BUFFER_COUNT as it was the requested amount
        for (unsigned int i = 0; i < BUFFER_COUNT; ++i) { 
            if (buffers[i].start && buffers[i].start != MAP_FAILED) {
                if (munmap(buffers[i].start, buffers[i].length) == -1) {
                    perror("munmap");
                }
            }
        }
        free(buffers);
        buffers = NULL;
    }
    if (fd != -1) {
        if (close(fd) == -1) {
            perror("close");
        }
        fd = -1;
    }
}

// Get terminal dimensions
void get_terminal_dimensions(int *width, int *height) {
    struct winsize ws;
    if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0 && ws.ws_row > 0) {
        *width = ws.ws_col;
        *height = ws.ws_row -1; // Use one less row to avoid issues with prompt
        if (*height <=0) *height = 1; // Ensure at least 1 row
    } else {
        // Fallback to defaults if ioctl fails or returns invalid sizes
        perror("TIOCGWINSZ failed or returned invalid dimensions, using defaults");
        *width = 80;  // Default width
        *height = 24; // Default height
    }
}


int main() {
    struct sigaction sa;
    sa.sa_handler = sigint_handler;
    sigemptyset(&sa.sa_mask);
    sa.sa_flags = 0; 
    if (sigaction(SIGINT, &sa, NULL) == -1) {
        perror("sigaction");
        return EXIT_FAILURE;
    }

    get_terminal_dimensions(&TERM_OUTPUT_WIDTH, &TERM_OUTPUT_HEIGHT);
    printf("Detected terminal size: %d columns, %d rows for video output.\n", TERM_OUTPUT_WIDTH, TERM_OUTPUT_HEIGHT);
    // Give user a moment to see the message before clearing screen
    sleep(1);


    if (init_device(DEFAULT_CAPTURE_WIDTH, DEFAULT_CAPTURE_HEIGHT) == -1) {
        fprintf(stderr, "Device initialization failed.\n");
        uninit_device(); 
        return EXIT_FAILURE;
    }

    if (start_capturing() == -1) {
        fprintf(stderr, "Failed to start capture.\n");
        uninit_device();
        return EXIT_FAILURE;
    }

    printf("Color ASCII Webcam started. Press Ctrl+C to stop.\n");
    printf("Ensure your terminal supports 24-bit true color and UTF-8 for '▀'.\n");
    
    main_loop(TERM_OUTPUT_WIDTH, TERM_OUTPUT_HEIGHT);

    // Restore cursor visibility and reset colors, just in case
    printf("\033[?25h\033[0m\n"); 
    printf("Cleaning up...\n");
    stop_capturing();
    uninit_device();

    printf("Color ASCII Webcam stopped.\n");
    return EXIT_SUCCESS;
}


