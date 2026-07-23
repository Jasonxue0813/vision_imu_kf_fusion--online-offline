/**
 * @file online_ypr_crop_node_runtime.cpp
 * @brief `OnlineYprCropNode` 的运行时消息与线程管理实现。
 *
 * 这里主要处理 fix / imu / image 三类消息的接收、缓存、
 * 处理线程控制、待处理帧调度，以及 `sp_lg` worker 的生命周期管理。
 */

#include "splg_fusion/app/online_ypr_crop_node.h"

// 功能：确保 `sp_lg` worker 进程已经启动。
void OnlineYprCropNode::ensureSpLgWorkerStarted() {
    if (!opt_.enable_sp_lg) return;
    if (sp_lg_worker_) return;

    sp_lg_worker_ = std::make_unique<SpLgWorkerClient>(sp_lg_executable_);
    sp_lg_worker_->start();
    std::cout << "[sp_lg] worker started\n";
}

// 功能：关闭 `sp_lg` worker 进程并清理相关资源。
void OnlineYprCropNode::shutdownSpLgWorker() {
    if (!sp_lg_worker_) return;
    sp_lg_worker_->stop();
    sp_lg_worker_.reset();
    std::cout << "[sp_lg] worker stopped\n";
}

// 功能：判断当前配置下是否需要保留按帧中间产物。
bool OnlineYprCropNode::shouldPersistFrameArtifacts() const {
    return opt_.persist_frame_artifacts;
}

// 功能：生成某一帧最终输出目录的路径。
std::filesystem::path OnlineYprCropNode::frameOutputDir(const std::string& frame_tag) const {
    return std::filesystem::path(opt_.out_dir) / frame_tag;
}

// 功能：生成某一帧工作目录或临时目录的路径。
std::filesystem::path OnlineYprCropNode::frameWorkingDir(const std::string& frame_tag) const {
    if (shouldPersistFrameArtifacts()) return frameOutputDir(frame_tag);
    return std::filesystem::temp_directory_path() / "splg_fusion_sp_lg" / frame_tag;
}

std::filesystem::path OnlineYprCropNode::resultImageDir() const {
    return std::filesystem::path(opt_.out_dir) / "result_img";
}

// 功能：把图像中间结果写到磁盘。
void OnlineYprCropNode::writeImageArtifact(const std::string& path, const cv::Mat& image) const {
    ensureParentDir(path);
    if (!cv::imwrite(path, image)) {
        throw std::runtime_error("Failed to write: " + path);
    }
}

void OnlineYprCropNode::writeResultImageArtifact(const std::string& file_name,
                                                 const cv::Mat& image) const {
    writeImageArtifact((resultImageDir() / file_name).string(), image);
}

void OnlineYprCropNode::copyResultImageArtifactIfExists(const std::string& src_path,
                                                        const std::string& file_name) const {
    if (src_path.empty() || !std::filesystem::exists(src_path)) return;
    const std::filesystem::path dst_path = resultImageDir() / file_name;
    ensureParentDir(dst_path.string());
    std::filesystem::copy_file(src_path,
                               dst_path,
                               std::filesystem::copy_options::overwrite_existing);
}

void OnlineYprCropNode::persistRawFrameIfNeeded(const SampledFrame& frame) const {
    if (!shouldPersistFrameArtifacts()) return;
    const std::string raw_path = (frameOutputDir(makeFrameTag(frame.image_index)) / "cam0_raw.png").string();
    writeImageArtifact(raw_path, frame.raw);
}

void OnlineYprCropNode::handleFixMessage(const sensor_msgs::NavSatFix& fix_msg) {
    if (stopped_) return;
    if (!std::isfinite(fix_msg.latitude) ||
        !std::isfinite(fix_msg.longitude) ||
        !std::isfinite(fix_msg.altitude)) {
        return;
    }
    const auto fix_receive_steady = std::chrono::steady_clock::now();
    const double t = fix_msg.header.stamp.toSec();
    if (t <= 0.0) return;

    bool should_append_fix_row = false;
    OnlineYprCropNode::FixFusionRow fix_row;
    auto fix_compute_start_steady = fix_receive_steady;
    auto fix_compute_end_steady = fix_receive_steady;

    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        heading_est_.update(t, fix_msg.latitude, fix_msg.longitude);
        const FixSample fix_sample{t, fix_msg.latitude, fix_msg.longitude, fix_msg.altitude};
        fixes_.push_back(fix_sample);
        all_fixes_.push_back(fix_sample);
        updateFusionOpenStateLocked(fix_sample);
        ensureContinuousFusionReferenceLocked(GeoPoint{fix_sample.lon, fix_sample.lat, fix_sample.alt});
        bootstrapFusionAnchorFromFixLocked(fix_sample);
        ++fix_count_;
        trimFixHistoryLocked(t);

        should_append_fix_row = true;
        fix_compute_start_steady = std::chrono::steady_clock::now();
        buildOnlineFixFusionRowLocked(fix_sample, all_fixes_.size(), fix_row);
        fix_compute_end_steady = std::chrono::steady_clock::now();
    }

    if (should_append_fix_row) {
        fix_row.fix_compute_latency_ms = computeElapsedMs(fix_compute_start_steady, fix_compute_end_steady);
        const auto fusion_ready_steady = std::chrono::steady_clock::now();
        fix_row.fix_topic_to_fusion_result_ms = computeElapsedMs(fix_receive_steady, fusion_ready_steady);
        appendFixFusionCsvRow(fix_row);
        if (online_mode_) {
            publishFixFusionRow(fix_row);
        }
        fix_fusion_csv_written_ = true;
    }

    state_cv_.notify_one();
}

void OnlineYprCropNode::handleImuMessage(const sensor_msgs::Imu& imu_msg) {
    if (stopped_) return;
    const auto imu_receive_steady = std::chrono::steady_clock::now();
    const double t = imu_msg.header.stamp.toSec();
    if (t <= 0.0) return;
    if (!std::isfinite(imu_msg.angular_velocity.x) ||
        !std::isfinite(imu_msg.angular_velocity.y) ||
        !std::isfinite(imu_msg.angular_velocity.z) ||
        !std::isfinite(imu_msg.linear_acceleration.x) ||
        !std::isfinite(imu_msg.linear_acceleration.y) ||
        !std::isfinite(imu_msg.linear_acceleration.z)) {
        return;
    }

    bool should_append_continuous_row = false;
    ContinuousFusionRow continuous_row;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        imu_tracker_.addSample(ImuSample{
            t,
            cv::Vec3d(imu_msg.angular_velocity.x, imu_msg.angular_velocity.y, imu_msg.angular_velocity.z),
            cv::Vec3d(imu_msg.linear_acceleration.x, imu_msg.linear_acceleration.y, imu_msg.linear_acceleration.z)});
        ++imu_count_;
        if (isFusionOpenLocked()) {
            should_append_continuous_row =
                predictContinuousFusionLocked(t, "imu_predict", &continuous_row);
        }
    }

    if (should_append_continuous_row) {
        continuous_row.output_latency_ms =
            computeElapsedMs(imu_receive_steady, std::chrono::steady_clock::now());
        appendContinuousFusionCsvRowThreadSafe(continuous_row);
    }

    state_cv_.notify_one();
}

void OnlineYprCropNode::handleImageMessage(const sensor_msgs::Image& image_msg) {
    if (stopped_) return;
    const auto topic_receive_steady = std::chrono::steady_clock::now();
    if (image_msg.header.stamp.toSec() <= 0.0) return;

    size_t frame_index = 0;
    bool should_process = false;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        ++image_count_;
        frame_index = image_count_;
        if (opt_.end_image_index > 0 && frame_index > static_cast<size_t>(opt_.end_image_index)) {
            return;
        }
        if (!isFusionOpenLocked()) {
            return;
        }

        const size_t start_index = (opt_.start_image_index > 0)
            ? static_cast<size_t>(opt_.start_image_index)
            : 1;
        if (frame_index >= start_index) {
            should_process = true;
            ++sampled_count_;
        }
    }

    if (!should_process) return;

    SampledFrame sampled_frame{
        frame_index,
        image_msg.header.stamp,
        topic_receive_steady,
        imageMsgToMat(image_msg)};
    persistRawFrameIfNeeded(sampled_frame);

    SampledFrame dropped_frame;
    bool dropped_incoming_frame = false;
    size_t buffered_frame_count = 0;
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (!processing_frame_active_ && pending_frames_.empty()) {
            pending_frames_.push_back(std::move(sampled_frame));
        } else {
            dropped_frame = std::move(sampled_frame);
            dropped_incoming_frame = true;
        }
        buffered_frame_count = pending_frames_.size() + (processing_frame_active_ ? 1u : 0u);
        if (buffered_frame_count > max_pending_frames_observed_) {
            max_pending_frames_observed_ = buffered_frame_count;
            if (buffered_frame_count <= 2) {
                ROS_WARN_STREAM("[VisionSerial] buffered frames=" << buffered_frame_count
                                << " (active=" << (processing_frame_active_ ? 1 : 0)
                                << ", pending=" << pending_frames_.size() << ")"
                                << " accepted=" << sampled_count_
                                << " seen=" << image_count_);
            }
        }
    }

    if (dropped_incoming_frame) {
        FrameRecord dropped_record = makeFrameRecordTemplate(dropped_frame);
        dropped_record.processing_start_steady = std::chrono::steady_clock::now();
        dropped_record.sync_wait_ms =
            computeElapsedMs(dropped_record.topic_receive_steady, dropped_record.processing_start_steady);
        dropped_record.failure_reason = "dropped_while_visual_busy";
        finalizeRecord(std::move(dropped_record));
        ROS_INFO_STREAM("[VisionSerial] dropped incoming frame image_index="
                        << frame_index
                        << " because visual solver is busy or another frame is already pending.");
    }

    state_cv_.notify_one();
}

void OnlineYprCropNode::fixCallback(const sensor_msgs::NavSatFix::ConstPtr& fix_msg) {
    if (stopped_) return;
    try {
        if (!fix_msg) return;
        handleFixMessage(*fix_msg);
    } catch (const std::exception& e) {
        shutdownWithError(e.what());
    }
}

void OnlineYprCropNode::imuCallback(const sensor_msgs::Imu::ConstPtr& imu_msg) {
    if (stopped_) return;
    try {
        if (!imu_msg) return;
        handleImuMessage(*imu_msg);
    } catch (const std::exception& e) {
        shutdownWithError(e.what());
    }
}

void OnlineYprCropNode::imageCallback(const sensor_msgs::Image::ConstPtr& image_msg) {
    if (stopped_) return;
    try {
        if (!image_msg) return;
        handleImageMessage(*image_msg);
    } catch (const std::exception& e) {
        shutdownWithError(e.what());
    }
}

void OnlineYprCropNode::trimFixHistoryLocked(double newest_t) {
    while (fixes_.size() > 2 && newest_t - fixes_[1].t > fix_history_keep_sec_) {
        fixes_.pop_front();
    }
}

void OnlineYprCropNode::startProcessingThread() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    if (processing_thread_.joinable()) return;
    processing_stop_requested_ = false;
    processing_thread_ = std::thread(&OnlineYprCropNode::processingLoop, this);
}

// 功能：请求后台处理线程停止。
void OnlineYprCropNode::requestProcessingStop() {
    {
        std::lock_guard<std::mutex> lock(state_mutex_);
        processing_stop_requested_ = true;
    }
    state_cv_.notify_all();
}

void OnlineYprCropNode::stopProcessingThread() {
    requestProcessingStop();
    if (processing_thread_.joinable()) {
        processing_thread_.join();
    }
}

void OnlineYprCropNode::processingLoop() {
    while (true) {
        SampledFrame frame;
        OnlineYprCropNode::PreparedFrameState prepared;
        FrameRecord record;
        const OnlineYprCropNode::PendingFrameAction action = waitForNextFrameAction(frame, prepared, record);
        if (action == OnlineYprCropNode::PendingFrameAction::kStop) break;

        try {
            executePendingFrameAction(action, frame, prepared, record);
        } catch (const std::exception& e) {
            shutdownWithError(e.what());
            break;
        }
    }
}

// 功能：执行一条待处理的帧动作。
void OnlineYprCropNode::executePendingFrameAction(OnlineYprCropNode::PendingFrameAction action,
                               const SampledFrame& frame,
                               const OnlineYprCropNode::PreparedFrameState& prepared,
                               FrameRecord& record) {
    if (action == OnlineYprCropNode::PendingFrameAction::kDrop) {
        finalizeRecord(record);
    } else if (action == OnlineYprCropNode::PendingFrameAction::kProcess) {
        try {
            processFrame(frame, prepared, record);
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(state_mutex_);
                processing_frame_active_ = false;
            }
            state_cv_.notify_one();
            throw;
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            processing_frame_active_ = false;
        }
        state_cv_.notify_one();
    }
}

void OnlineYprCropNode::drainReadyFramesInline(bool flush_pending_on_shutdown) {
    while (true) {
        SampledFrame frame;
        OnlineYprCropNode::PreparedFrameState prepared;
        FrameRecord record;
        OnlineYprCropNode::PendingFrameAction action = OnlineYprCropNode::PendingFrameAction::kWait;

        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (pending_frames_.empty()) return;

            action = prepareFrontFrameLocked(frame, prepared, record);
            if (action == OnlineYprCropNode::PendingFrameAction::kWait) {
                if (!flush_pending_on_shutdown) return;

                const SampledFrame& stalled_frame = pending_frames_.front();
                record = makeFrameRecordTemplate(stalled_frame);
                record.processing_start_steady = std::chrono::steady_clock::now();
                record.sync_wait_ms =
                    computeElapsedMs(record.topic_receive_steady, record.processing_start_steady);
                if (tryFillGpsLocked(record.image_t, record.gps)) {
                    record.failure_reason = "waiting_for_future_imu_on_shutdown";
                    ++reject_stats_.imu_query_failed;
                } else {
                    record.failure_reason = "waiting_for_future_gps_on_shutdown";
                    ++reject_stats_.gps_interp_failed;
                }
                pending_frames_.pop_front();
                action = OnlineYprCropNode::PendingFrameAction::kDrop;
            } else if (action == OnlineYprCropNode::PendingFrameAction::kProcess || action == OnlineYprCropNode::PendingFrameAction::kDrop) {
                record.processing_start_steady = std::chrono::steady_clock::now();
                record.sync_wait_ms =
                    computeElapsedMs(record.topic_receive_steady, record.processing_start_steady);
            }
        }

        if (action == OnlineYprCropNode::PendingFrameAction::kStop || action == OnlineYprCropNode::PendingFrameAction::kWait) return;
        executePendingFrameAction(action, frame, prepared, record);
    }
}

OnlineYprCropNode::PendingFrameAction OnlineYprCropNode::waitForNextFrameAction(SampledFrame& frame,
                                          OnlineYprCropNode::PreparedFrameState& prepared,
                                          FrameRecord& record) {
    std::unique_lock<std::mutex> lock(state_mutex_);
    for (;;) {
        if (pending_frames_.empty()) {
            if (processing_stop_requested_) return OnlineYprCropNode::PendingFrameAction::kStop;
            state_cv_.wait(lock, [this]() {
                return processing_stop_requested_ || !pending_frames_.empty();
            });
            continue;
        }

        const OnlineYprCropNode::PendingFrameAction action = prepareFrontFrameLocked(frame, prepared, record);
        if (action == OnlineYprCropNode::PendingFrameAction::kProcess || action == OnlineYprCropNode::PendingFrameAction::kDrop) {
            record.processing_start_steady = std::chrono::steady_clock::now();
            record.sync_wait_ms =
                computeElapsedMs(record.topic_receive_steady, record.processing_start_steady);
            return action;
        }

        if (processing_stop_requested_) {
            const SampledFrame& stalled_frame = pending_frames_.front();
            record = makeFrameRecordTemplate(stalled_frame);
            record.processing_start_steady = std::chrono::steady_clock::now();
            record.sync_wait_ms =
                computeElapsedMs(record.topic_receive_steady, record.processing_start_steady);
            if (tryFillGpsLocked(record.image_t, record.gps)) {
                record.failure_reason = "waiting_for_future_imu_on_shutdown";
                ++reject_stats_.imu_query_failed;
            } else {
                record.failure_reason = "waiting_for_future_gps_on_shutdown";
                ++reject_stats_.gps_interp_failed;
            }
            ROS_WARN_STREAM("[SyncBuffer] dropping pending frame image_index="
                            << stalled_frame.image_index
                            << " during shutdown because required future GPS/IMU samples never arrived."
                            << " image_t=" << record.image_t
                            << " imu_query_t=" << record.imu_query_t);
            pending_frames_.pop_front();
            return OnlineYprCropNode::PendingFrameAction::kDrop;
        }

        state_cv_.wait(lock);
    }
}

OnlineYprCropNode::PendingFrameAction OnlineYprCropNode::prepareFrontFrameLocked(SampledFrame& frame,
                                           OnlineYprCropNode::PreparedFrameState& prepared,
                                           FrameRecord& record) {
    if (pending_frames_.empty()) return OnlineYprCropNode::PendingFrameAction::kWait;

    const SampledFrame& front = pending_frames_.front();
    record = makeFrameRecordTemplate(front);
    const double image_t = front.stamp.toSec();
    const double imu_query_t = image_t + ext_.cam_to_imu_shift_s;

    if (fixes_.empty() || fixes_.back().t < image_t) return OnlineYprCropNode::PendingFrameAction::kWait;
    if (!imu_tracker_.initialized() || imu_tracker_.latestTime() < imu_query_t) {
        return OnlineYprCropNode::PendingFrameAction::kWait;
    }

    if (image_t < fixes_.front().t) {
        ++reject_stats_.gps_interp_failed;
        record.failure_reason = "image_before_oldest_fix";
        pending_frames_.pop_front();
        return OnlineYprCropNode::PendingFrameAction::kDrop;
    }
    if (imu_query_t < imu_tracker_.startTime()) {
        ++reject_stats_.imu_query_failed;
        tryFillGpsLocked(image_t, record.gps);
        record.failure_reason = "imu_query_before_tracker_start";
        pending_frames_.pop_front();
        return OnlineYprCropNode::PendingFrameAction::kDrop;
    }

    prepared.image_t = image_t;
    prepared.imu_query_t = imu_query_t;
    if (tryFillGpsLocked(image_t, prepared.gps)) {
        record.gps = prepared.gps;
    }
    if (!isFusionOpenLocked()) {
        record.failure_reason = std::isfinite(fusion_initial_altitude_)
            ? "fusion_not_open_height_not_reached"
            : "fusion_not_open_initializing_altitude";
        pending_frames_.pop_front();
        return OnlineYprCropNode::PendingFrameAction::kDrop;
    }
    const bool heading_found = heading_est_.query(image_t, prepared.heading);
    if (!heading_found || !prepared.heading.valid) {
        ++reject_stats_.heading_invalid;
        record.failure_reason = classifyHeadingFailureReason(heading_found, prepared.heading);
        pending_frames_.pop_front();
        return OnlineYprCropNode::PendingFrameAction::kDrop;
    }
    if (!std::isfinite(prepared.gps.lon) ||
        !std::isfinite(prepared.gps.lat) ||
        !std::isfinite(prepared.gps.alt)) {
        ++reject_stats_.gps_interp_failed;
        record.failure_reason = "gps_interp_failed";
        pending_frames_.pop_front();
        return OnlineYprCropNode::PendingFrameAction::kDrop;
    }
    if (!imu_tracker_.query(imu_query_t,
                            prepared.imu_raw_roll_deg,
                            prepared.imu_raw_pitch_deg)) {
        ++reject_stats_.imu_query_failed;
        record.failure_reason = "imu_query_failed";
        pending_frames_.pop_front();
        return OnlineYprCropNode::PendingFrameAction::kDrop;
    }

    frame = std::move(pending_frames_.front());
    pending_frames_.pop_front();
    processing_frame_active_ = true;
    return OnlineYprCropNode::PendingFrameAction::kProcess;
}
