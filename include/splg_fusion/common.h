/**
 * @file common.h
 * @brief `splg_fusion` 公共依赖头。
 *
 * 这个文件统一收集工程里常用的 ROS、OpenCV、GDAL 以及 STL 头文件，
 * 让其他模块可以通过一个公共头拿到基础依赖。
 */

#pragma once

#include "global_heading.h"

#include <opencv2/opencv.hpp>

#include <gdal_priv.h>
#include <ogr_spatialref.h>

#include <ros/ros.h>
#include <ros/time.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Image.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <std_msgs/String.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <signal.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>
