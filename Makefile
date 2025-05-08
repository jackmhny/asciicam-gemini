 # Makefile for ascii_cam_opencv

CXX := g++
PKG_CONFIG := pkg-config

# Determine OpenCV pkg-config module: try opencv4, fallback to opencv
OPENCV_PKG := $(shell $(PKG_CONFIG) --exists opencv4 && echo opencv4 || echo opencv)

CXXFLAGS := -std=c++11 -O2 -Wall -Wextra $(shell $(PKG_CONFIG) --cflags $(OPENCV_PKG))
LDFLAGS := $(shell $(PKG_CONFIG) --libs $(OPENCV_PKG))

TARGET := ascii_cam_opencv
SRC := ascii_cam_opencv.cpp

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(SRC)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDFLAGS)

clean:
	rm -f $(TARGET)