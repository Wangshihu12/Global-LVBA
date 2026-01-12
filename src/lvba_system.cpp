#include "lvba_system.h"

namespace lvba {
    
LvbaSystem::LvbaSystem(ros::NodeHandle& nh) : nh_(nh)                                
{
    dataset_io_.reset(new DatasetIO(nh_));

    cloud_pub_after_ = nh_.advertise<sensor_msgs::PointCloud2>("/lvba/cloud_after", 1, true);
    cloud_pub_before_ = nh_.advertise<sensor_msgs::PointCloud2>("/lvba/cloud_before", 1, true);
    pub_test_ = nh_.advertise<sensor_msgs::PointCloud2>("/map_test", 100);
    pub_path_ = nh_.advertise<sensor_msgs::PointCloud2>("/map_path", 100);
    pub_show_ = nh_.advertise<sensor_msgs::PointCloud2>("/map_show", 100);
    pub_cute_ = nh_.advertise<sensor_msgs::PointCloud2>("/map_cute", 100);

    pub_cloud_before_ = nh_.advertise<sensor_msgs::PointCloud2>("viz/cloud_before", 1, true);
    pub_cloud_after_  = nh_.advertise<sensor_msgs::PointCloud2>("viz/cloud_after", 1, true);

    nh_.param<bool>("data_config/enable_lidar_ba", enable_lidar_ba_, true);
    nh_.param<bool>("data_config/enable_visual_ba", enable_visual_ba_, true);
    nh_.param<double>("track_fusion/min_view_angle", min_view_angle_deg_, 8.0);
    nh_.param<double>("track_fusion/reproj_mean_thr", reproj_mean_thr_px_, 3.0);
}

void LvbaSystem::runFullPipeline() 
{
    initFromDatasetIO();
    if(enable_lidar_ba_) runLidarBA();
    if(enable_visual_ba_) runVisualBAWithLidarAssist();
    ros::spin();
}

/**
 * [功能描述]：执行LiDAR辅助的视觉Bundle Adjustment优化流程。
 *            该函数整合了从LiDAR优化结果到视觉BA的完整处理管线，
 *            包括地图构建、位姿更新、深度生成、特征匹配、轨迹构建、位姿优化和可视化。
 * @return 无返回值
 */
void LvbaSystem::runVisualBAWithLidarAssist()
{
    // 步骤1：从LiDAR优化后的点云构建网格地图（用于后续深度生成）
    buildGridMapFromOptimized();
    
    // 步骤2：根据LiDAR优化结果更新相机位姿（LiDAR-Camera外参标定）
    updateCameraPosesFromLidar();
    
    // 步骤3：利用体素化点云为每帧图像生成深度图
    generateDepthWithVoxel();
    
    // 步骤4：使用GPU加速提取图像特征点并进行特征匹配
    extractAndMatchFeaturesGPU();
    
    // 步骤5：构建特征点轨迹（tracks）并融合生成3D地图点
    BuildTracksAndFuse3D();
    
    // 步骤6：执行视觉BA优化，精化相机位姿
    optimizeCameraPoses();
    
    // 步骤7：可视化特征点的重投影结果
    visualizeProj();
    
    // 步骤8：发布带RGB颜色的点云到ROS话题
    pubRGBCloud();
}

template <typename T>
void LvbaSystem::pub_pl_func(T &pl, ros::Publisher &pub)
{
  pl.height = 1; pl.width = pl.size();
  sensor_msgs::PointCloud2 output;
  pcl::toROSMsg(pl, output);
  output.header.frame_id = "map";
  output.header.stamp = ros::Time::now();
  pub.publish(output);
}

void LvbaSystem::data_show(vector<IMUST> x_buf, vector<pcl::PointCloud<PointType>::Ptr> &pl_fulls)
{
  IMUST es0 = x_buf[0];
  for(uint i=0; i<x_buf.size(); i++)
  {
    x_buf[i].p = es0.R.transpose() * (x_buf[i].p - es0.p);
    x_buf[i].R = es0.R.transpose() * x_buf[i].R;
  }

  pcl::PointCloud<PointType> pl_send, pl_path;
  int winsize = x_buf.size();
  for(int i=0; i<winsize; i++)
  {
    pcl::PointCloud<PointType> pl_tem = *pl_fulls[i];
    down_sampling_voxel(pl_tem, 0.05);
    pl_transform(pl_tem, x_buf[i]);
    pl_send += pl_tem;

    // if((i%2==0 && i!=0) || i == winsize-1)
    // {
    //   pub_pl_func(pl_send, pub_show_);
    //   pl_send.clear();
    //   sleep(0.3);
    // }

    PointType ap;
    ap.x = x_buf[i].p.x();
    ap.y = x_buf[i].p.y();
    ap.z = x_buf[i].p.z();
    ap.curvature = i;
    pl_path.push_back(ap);
  }
  down_sampling_voxel(pl_send, 0.05);
  pub_pl_func(pl_send, pub_show_);
  pub_pl_func(pl_path, pub_path_);
}

/**
 * [功能描述]：运行滑动窗口LiDAR BA优化，将连续帧聚合为锚点帧。
 *            通过在每个窗口内进行局部BA优化，然后将窗口内的点云合并为一个锚点点云，
 *            从而减少全局优化时的变量数量，提高计算效率。
 * @param x_buf_full：输入的所有帧位姿向量，每个元素为IMUST类型（包含R旋转矩阵和p平移向量）
 * @param pl_fulls_full：输入的所有帧点云向量，与x_buf_full一一对应
 * @param anchor_poses：输出的锚点帧位姿向量
 * @param anchor_clouds：输出的锚点帧合并点云向量
 * @return 无返回值，结果通过引用参数返回
 */
void LvbaSystem::runWindowBA(const std::vector<IMUST>& x_buf_full,
                             const std::vector<pcl::PointCloud<PointType>::Ptr>& pl_fulls_full,
                             std::vector<IMUST>& anchor_poses,
                             std::vector<pcl::PointCloud<PointType>::Ptr>& anchor_clouds)
{
    // 清空输出容器
    anchor_poses.clear();
    anchor_clouds.clear();

    // ========== 第一步：读取配置参数 ==========
    const bool run_window = dataset_io_->window_ba_enable_;        // 是否启用窗口BA
    const int window_size = dataset_io_->window_ba_size_;          // 每个窗口包含的帧数
    const double anchor_leaf = dataset_io_->anchor_leaf_size_;     // 锚点点云降采样的体素大小
    const bool use_window_ba_rel = dataset_io_->use_window_ba_rel_;// 是否使用窗口BA的相对位姿
    const int total_size = static_cast<int>(x_buf_full.size());    // 总帧数

    // ========== 第二步：处理不启用窗口BA的情况 ==========
    // 如果不启用窗口BA，则直接将每帧作为一个锚点帧
    if (!run_window) {
        anchor_poses = x_buf_full;
        anchor_clouds = pl_fulls_full;
        for (int i = 0; i < total_size; ++i) {
            anchor_index_per_frame_[i] = i;        // 每帧对应自身作为锚点
            rel_poses_to_anchor_[i].setZero();     // 相对位姿为单位变换（identity）
        }
        return;
    }

    printf("[WindowBA] Running Window LiDAR BA, window size=%d, anchor leaf=%.2f ...\n", window_size, anchor_leaf);

    // ========== 第三步：滑动窗口处理 ==========
    int win_total = 0;    // 总窗口数
    int win_skipped = 0;  // 跳过的窗口数（因体素数量不足）
    
    // 遍历所有帧，按窗口大小分组处理
    for (int start = 0; start < total_size; start += window_size) {
        int end = std::min(start + window_size, total_size);  // 当前窗口的结束索引
        printProgressBar(end, total_size);  // 打印进度条

        int curr_win = end - start;  // 当前窗口的实际帧数
        if (curr_win <= 0) break;
        ++win_total;

        // ---------- 3.1 提取当前窗口的位姿和点云 ----------
        // x_win: 当前窗口内帧的位姿（将被优化修改）
        std::vector<IMUST> x_win(x_buf_full.begin() + start, x_buf_full.begin() + end);
        // x_win_odom: 保存原始里程计位姿（不被修改）
        std::vector<IMUST> x_win_odom = x_win;
        // x_win_aligned: 对齐后的位姿
        std::vector<IMUST> x_win_aligned = x_win;
        // pl_win: 当前窗口内帧的点云
        std::vector<pcl::PointCloud<PointType>::Ptr> pl_win;
        pl_win.reserve(curr_win);
        for (int i = start; i < end; ++i) pl_win.push_back(pl_fulls_full[i]);

        // ---------- 3.2 构建体素地图并进行窗口内BA优化 ----------
        // surf_map: 体素地图，用于存储点云的体素化结构
        std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
        for (int j = 0; j < curr_win; ++j) {
            // 对窗口内每帧点云进行体素切割
            cut_voxel(surf_map, *pl_win[j], x_win[j], j, curr_win,
                      dataset_io_->stage1_root_voxel_size_, dataset_io_->stage1_eigen_ratio_array_[0]);
        }

        // 创建BALM2优化器和体素Hessian矩阵
        std::unique_ptr<BALM2> opt_lsv(new BALM2(curr_win));
        std::unique_ptr<VOX_HESS> voxhess(new VOX_HESS(curr_win));
        
        // 遍历体素地图，执行重切割和优化准备
        for (auto iter = surf_map.begin(); iter != surf_map.end() && nh_.ok(); ++iter) {
            iter->second->recut(x_win);       // 根据当前位姿重新切割体素
            iter->second->tras_opt(*voxhess); // 将体素信息传递给Hessian矩阵
        }
        
        // 检查有效体素数量是否足够（至少需要3倍于帧数的体素）
        if (voxhess->plvec_voxels.size() < static_cast<size_t>(3 * x_win.size())) {
            for (auto& kv : surf_map) delete kv.second;
            ++win_skipped;
            continue;  // 体素不足，跳过当前窗口
        }
        
        // 执行阻尼迭代优化，更新x_win中的位姿
        opt_lsv->damping_iter(x_win, *voxhess);

        // 释放体素地图内存
        for (auto& kv : surf_map) delete kv.second;

        // ---------- 3.3 位姿对齐：将优化后的位姿对齐到原始里程计坐标系 ----------
        if (use_window_ba_rel && !x_win.empty()) {
            // 计算对齐变换：使优化后的第一帧与原始第一帧对齐
            const IMUST& odom0 = x_win_odom[0];  // 原始里程计的第一帧位姿
            const IMUST& opt0  = x_win[0];       // 优化后的第一帧位姿
            
            // R_align: 对齐旋转矩阵，形状为 (3, 3)
            // 公式: R_align = R_odom0 * R_opt0^T
            Eigen::Matrix3d R_align = odom0.R * opt0.R.transpose();
            // p_align: 对齐平移向量，形状为 (3,)
            // 公式: p_align = p_odom0 - R_align * p_opt0
            Eigen::Vector3d p_align = odom0.p - R_align * opt0.p;
            
            // 将对齐变换应用到窗口内所有帧
            for (int j = 0; j < curr_win; ++j) {
                x_win_aligned[j].R = R_align * x_win[j].R;
                x_win_aligned[j].p = R_align * x_win[j].p + p_align;
            }
        } else {
            // 不使用相对位姿优化，直接使用原始里程计位姿
            x_win_aligned = x_win_odom;
        }

        // ---------- 3.4 合并窗口内点云为锚点点云 ----------
        pcl::PointCloud<PointType>::Ptr merged(new pcl::PointCloud<PointType>());
        const IMUST anchor_pose = x_win_odom[0];  // 锚点位姿取窗口第一帧的原始位姿
        const int anchor_idx = static_cast<int>(anchor_poses.size());  // 当前锚点的索引
        
        for (int j = 0; j < curr_win; ++j) {
            pcl::PointCloud<PointType> tmp = *pl_win[j];
            
            // 计算当前帧相对于锚点帧的相对位姿
            IMUST rel;
            // rel.R: 相对旋转矩阵 = R_anchor^T * R_current，形状为 (3, 3)
            rel.R = anchor_pose.R.transpose() * x_win_aligned[j].R;
            // rel.p: 相对平移向量 = R_anchor^T * (p_current - p_anchor)，形状为 (3,)
            rel.p = anchor_pose.R.transpose() * (x_win_aligned[j].p - anchor_pose.p);
            
            // 将点云从当前帧坐标系变换到锚点帧坐标系
            pl_transform(tmp, rel);
            *merged += tmp;  // 累加到合并点云

            // 记录全局索引与锚点的对应关系
            const int global_idx = start + j;
            if (global_idx < total_size) {
                rel_poses_to_anchor_[global_idx] = rel;        // 保存相对位姿
                anchor_index_per_frame_[global_idx] = anchor_idx;  // 保存锚点索引
            }
        }
        
        // 对合并后的点云进行体素降采样
        down_sampling_voxel2(*merged, anchor_leaf);

        // 保存锚点帧的位姿和点云
        anchor_poses.push_back(anchor_pose);
        anchor_clouds.push_back(merged);
    }
    std::cout << std::endl;

    // ========== 第四步：输出统计信息 ==========
    if (win_total > 0) {
        printf("[WindowBA] skipped %d/%d windows (%.2f%%)\n",
               win_skipped, win_total,
               100.0 * static_cast<double>(win_skipped) / static_cast<double>(win_total));
    }
}

/**
 * [功能描述]：执行全局LiDAR Bundle Adjustment（BA）优化。
 *            该函数采用两阶段优化策略，通过体素化点云并使用BALM2算法对锚点帧位姿进行优化，
 *            然后将优化结果传播到所有原始帧。
 * @return 无返回值
 */
void LvbaSystem::runLidarBA() 
{
    // ========== 第一步：获取输入数据 ==========
    // x_buf_full: 所有帧的位姿（包含旋转R和平移p）
    std::vector<IMUST> x_buf_full = dataset_io_->x_buf_;
    // pl_fulls_full: 所有帧对应的完整点云
    std::vector<pcl::PointCloud<PointType>::Ptr> pl_fulls_full = dataset_io_->pl_fulls_;
    const int total_size = static_cast<int>(x_buf_full.size());
    
    // 检查是否有有效数据
    if (total_size == 0) {
        ROS_WARN("No poses in buffer, skip runLidarBA.");
        return;
    }

    // ========== 第二步：可视化数据并等待用户确认 ==========
    data_show(x_buf_full, pl_fulls_full);
    printf("If no problem, input '1' to continue or '0' to exit...\n");
    int cont_flag = 1;
    std::cin >> cont_flag;
    if (cont_flag == 0) {
        return;  // 用户选择退出
    }

    // ========== 第三步：初始化锚点帧相关数据结构 ==========
    std::vector<IMUST> anchor_poses;                              // 锚点帧的位姿
    std::vector<pcl::PointCloud<PointType>::Ptr> anchor_clouds;   // 锚点帧的点云

    // rel_poses_to_anchor_: 每个原始帧相对于其锚点帧的相对位姿
    rel_poses_to_anchor_.assign(total_size, IMUST());
    // anchor_index_per_frame_: 记录每个原始帧对应的锚点帧索引，-1表示无效
    anchor_index_per_frame_.assign(total_size, -1);

    // ========== 第四步：运行窗口BA，生成锚点帧 ==========
    // 通过滑动窗口BA将多帧聚合为锚点帧，减少优化变量数量
    runWindowBA(x_buf_full, pl_fulls_full, anchor_poses, anchor_clouds);

    // ========== 第五步：配置两阶段优化参数 ==========
    const int win_size = static_cast<int>(anchor_poses.size());  // 锚点帧数量
    const char* pass_name[2] = {"Stage 1", "Stage 2"};           // 阶段名称
    bool run_stage1 = dataset_io_->stage1_enable_;               // 是否启用第一阶段

    // 两阶段的体素大小配置
    double root_voxel_size[2];
    // 两阶段的特征值比例阈值数组（用于平面特征判断）
    std::array<float, 4> eigen_ratio_array[2];

    root_voxel_size[0] = dataset_io_->stage1_root_voxel_size_;  // 第一阶段体素大小
    root_voxel_size[1] = dataset_io_->stage2_root_voxel_size_;  // 第二阶段体素大小
    eigen_ratio_array[0].fill(0.f);
    eigen_ratio_array[1].fill(0.f);

    // 从配置中读取特征值比例阈值
    for (size_t i = 0; i < eigen_ratio_array[0].size(); ++i) {
        if (i < dataset_io_->stage1_eigen_ratio_array_.size())
            eigen_ratio_array[0][i] = dataset_io_->stage1_eigen_ratio_array_[i];
        if (i < dataset_io_->stage2_eigen_ratio_array_.size())
            eigen_ratio_array[1][i] = dataset_io_->stage2_eigen_ratio_array_[i];
    }

    // ========== 第六步：执行两阶段全局BA优化 ==========
    // 根据配置决定从哪个阶段开始（0=Stage1, 1=Stage2）
    int start_idx = run_stage1 ? 0 : 1;
    for (int idx = start_idx; idx < 2; ++idx) {
        cout << "[runLidarBA] Global LiDAR BA start... " << pass_name[idx] << endl;

        // 设置当前阶段的特征值比例阈值
        set_eigen_ratio_array(eigen_ratio_array[idx]);
        
        // surf_map: 体素地图，键为体素位置，值为八叉树根节点
        std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
        pcl::PointCloud<PointType> pl_send;  // 用于可视化的点云
        pub_pl_func(pl_send, pub_show_);

        // 对所有锚点帧的点云进行体素化切割
        float eigen_ratio_val = eigen_ratio_array[idx][0];
        for (int j = 0; j < win_size; j++) {
            // cut_voxel: 将点云分割到体素中，构建体素地图
            cut_voxel(surf_map, *anchor_clouds[j], anchor_poses[j], j, win_size,
                      root_voxel_size[idx], eigen_ratio_val);
        }

        // 创建BALM2优化器和体素Hessian矩阵
        std::unique_ptr<BALM2> opt_lsv(new BALM2(win_size));
        std::unique_ptr<VOX_HESS> voxhess(new VOX_HESS(win_size));

        // 遍历所有体素，执行重切割、优化准备和可视化
        for (auto iter = surf_map.begin(); iter != surf_map.end() && nh_.ok(); ++iter) {
            iter->second->recut(anchor_poses);                      // 根据当前位姿重新切割体素
            iter->second->tras_opt(*voxhess);                       // 将体素信息传递给优化器
            iter->second->tras_display(pl_send, anchor_poses, 0);   // 生成可视化点云
        }

        // 对可视化点云进行降采样并发布
        down_sampling_voxel(pl_send, 0.05);
        pub_pl_func(pl_send, pub_cute_);

        pl_send.clear();
        pub_pl_func(pl_send, pub_cute_);

        // 执行阻尼迭代优化，更新anchor_poses中的位姿
        opt_lsv->damping_iter(anchor_poses, *voxhess);

        // 释放体素地图内存
        for (auto& kv : surf_map) delete kv.second;
    }

    cout << "[runLidarBA] Global BALM Finish..." << endl;

    // ========== 第七步：将锚点帧优化结果传播到所有原始帧 ==========
    optimized_x_buf_ = x_buf_full;  // 复制原始位姿作为基础
    for (size_t idx = 0; idx < optimized_x_buf_.size(); ++idx) {
        // 获取当前帧对应的锚点帧索引
        const int anchor_idx = (idx < anchor_index_per_frame_.size()) ? anchor_index_per_frame_[idx] : -1;
        if (anchor_idx < 0 || anchor_idx >= static_cast<int>(anchor_poses.size())) continue;

        // 获取相对位姿和优化后的锚点位姿
        const IMUST& rel = rel_poses_to_anchor_[idx];    // 当前帧相对于锚点帧的相对位姿
        const IMUST& anchor = anchor_poses[anchor_idx];  // 优化后的锚点帧位姿

        // 通过位姿复合计算优化后的帧位姿
        // 公式: T_world_frame = T_world_anchor * T_anchor_frame
        IMUST& out = optimized_x_buf_[idx];
        out.R = anchor.R * rel.R;                // 旋转矩阵复合
        out.p = anchor.R * rel.p + anchor.p;     // 平移向量复合
    }

    // 将优化结果写回dataset_io_
    dataset_io_->x_buf_ = optimized_x_buf_;
    
    // 可视化优化后的结果
    // data_show(anchor_poses, anchor_clouds);
    data_show(dataset_io_->x_buf_, dataset_io_->pl_fulls_);
}

/**
 * [功能描述]：根据LiDAR BA优化结果更新相机位姿。
 *            核心思想是计算LiDAR优化前后的位姿变化量（delta），
 *            然后将该变化量应用到原始相机位姿上，从而使相机位姿与LiDAR优化结果保持一致。
 * @return 无返回值，更新后的相机位姿存储在成员变量poses_中
 */
void LvbaSystem::updateCameraPosesFromLidar()
{
    // 获取LiDAR和相机的位姿数据
    const auto& lidar_opt = dataset_io_->x_buf_;          // LiDAR优化后的位姿
    const auto& lidar_orig = dataset_io_->x_buf_before_;  // LiDAR优化前的原始位姿
    const auto& cam_orig = dataset_io_->image_poses_;     // 相机原始位姿（SE3类型）

    // 清空并预分配输出容器
    poses_.clear();
    poses_.reserve(cam_orig.size());

    // ========== 第一步：提取LiDAR帧的时间戳列表 ==========
    std::vector<double> ts;
    ts.reserve(lidar_opt.size());
    for (const auto& x : lidar_opt) ts.push_back(x.t);

    // ========== 第二步：遍历每张图像，更新其相机位姿 ==========
    for (size_t i = 0; i < images_ids_.size(); ++i) {
        double t_img = images_ids_[i];  // 当前图像的时间戳

        // ---------- 2.1 查找时间上最近的LiDAR帧 ----------
        // 使用二分查找定位第一个 >= t_img 的LiDAR帧
        auto it = std::lower_bound(ts.begin(), ts.end(), t_img);
        size_t idx = (it == ts.end()) ? ts.size() - 1 : static_cast<size_t>(it - ts.begin());
        
        // 比较前后两帧，选择时间差更小的那个
        if (it != ts.begin() && it != ts.end()) {
            size_t prev = idx - 1;
            if (std::abs(ts[prev] - t_img) < std::abs(ts[idx] - t_img)) idx = prev;
        }
        
        // 索引越界检查，若无效则保留原始相机位姿
        if (idx >= lidar_opt.size() || idx >= lidar_orig.size()) {
            poses_.push_back(cam_orig[i]);
            continue;
        }

        // ---------- 2.2 计算LiDAR优化前后的位姿变化量 ----------
        // T_opt: 优化后的LiDAR位姿（SE3变换矩阵）
        Sophus::SE3 T_opt(lidar_opt[idx].R, lidar_opt[idx].p);
        // T_orig: 优化前的LiDAR位姿（SE3变换矩阵）
        Sophus::SE3 T_orig(lidar_orig[idx].R, lidar_orig[idx].p);
        // T_delta: 位姿变化量，公式: T_delta = T_opt * T_orig^(-1)
        // 表示从原始位姿到优化位姿的增量变换
        Sophus::SE3 T_delta = T_opt * T_orig.inverse();

        // ---------- 2.3 将位姿变化量应用到相机位姿 ----------
        // T_cam_new: 更新后的相机位姿
        // 公式: T_cam_new = T_delta * T_cam_orig
        // 即将LiDAR的优化增量传递给相机
        Sophus::SE3 T_cam_new = T_delta * cam_orig[i];
        poses_.push_back(T_cam_new);
    }
}

/**
 * [功能描述]：从DatasetIO对象中初始化LvbaSystem的成员变量，
 *            包括数据集路径、图像信息、相机内参、以及各坐标系之间的外参变换矩阵。
 * @return 无返回值
 */
void LvbaSystem::initFromDatasetIO() {

    // ========== 第一步：从dataset_io_中复制基本数据 ==========
    dataset_path_  = dataset_io_->dataset_path_;   // 数据集路径
    images_ids_    = dataset_io_->images_ids_;     // 图像ID列表
    poses_before_  = dataset_io_->image_poses_;    // 优化前的图像位姿
    // poses_         = dataset_io_->image_poses_;
    // cloud_         = dataset_io_->cloud_;
    all_voxel_ids_ = dataset_io_->all_voxel_ids_;  // 所有体素ID

    // 检查图像数量与位姿数量是否匹配
    if (images_ids_.size() != poses_before_.size()) {
        std::cerr << "Error: Number of images and poses do not match!" << std::endl;
        return;
    }

    // ========== 第二步：生成所有图像对的组合 ==========
    // 使用双重循环生成所有不重复的图像对 (i, j)，其中 i < j
    // 用于后续的图像匹配
    for (size_t i = 0; i < images_ids_.size(); ++i) {
        for (size_t j = i + 1; j < images_ids_.size(); ++j) {
            image_pairs_.push_back(std::make_pair(images_ids_[i], images_ids_[j]));
        }
    }

    // ========== 第三步：初始化关键点和深度容器 ==========
    all_keypoints_.resize(images_ids_.size());   // 预分配关键点容器大小
    all_depths_.reserve(images_ids_.size());     // 预留深度数据容器容量

    // ========== 第四步：设置图像尺寸和缩放比例 ==========
    image_width_  = dataset_io_->width_;         // 图像宽度
    image_height_ = dataset_io_->height_;        // 图像高度
    scale_ = dataset_io_->resize_scale_;         // 图像缩放比例

    // ========== 第五步：设置相机内参 ==========
    fx_ = dataset_io_->fx_;  // 焦距x
    fy_ = dataset_io_->fy_;  // 焦距y
    cx_ = dataset_io_->cx_;  // 主点x坐标
    cy_ = dataset_io_->cy_;  // 主点y坐标
    d0_ = dataset_io_->k1_;  // 径向畸变系数k1
    d1_ = dataset_io_->k2_;  // 径向畸变系数k2
    d2_ = dataset_io_->p1_;  // 切向畸变系数p1
    d3_ = dataset_io_->p2_;  // 切向畸变系数p2

    // ========== 第六步：计算Lidar到Camera的外参变换 ==========
    // 获取Lidar到Camera的平移向量和旋转矩阵
    std::vector<double> t_lidar2cam = dataset_io_->cameraextrinT_;
    std::vector<double> r_lidar2cam = dataset_io_->cameraextrinR_;

    // tcl_: Lidar坐标系到Camera坐标系的平移向量，形状为 (3,)
    tcl_ << t_lidar2cam[0], t_lidar2cam[1], t_lidar2cam[2];
    // Rcl_: Lidar坐标系到Camera坐标系的旋转矩阵，形状为 (3, 3)
    Rcl_ << r_lidar2cam[0], r_lidar2cam[1], r_lidar2cam[2],
            r_lidar2cam[3], r_lidar2cam[4], r_lidar2cam[5],
            r_lidar2cam[6], r_lidar2cam[7], r_lidar2cam[8];

    // ========== 第七步：计算Lidar到IMU以及Camera到IMU的外参变换 ==========
    // 获取Lidar到IMU的平移向量和旋转矩阵
    std::vector<double> t_lidar2imu = dataset_io_->extrinT_;
    std::vector<double> r_lidar2imu = dataset_io_->extrinR_;

    // til: IMU坐标系到Lidar坐标系的平移向量，形状为 (3,)
    // Ril: IMU坐标系到Lidar坐标系的旋转矩阵，形状为 (3, 3)
    Eigen::Vector3d til; Eigen::Matrix3d Ril;
    til << t_lidar2imu[0], t_lidar2imu[1], t_lidar2imu[2];
    Ril << r_lidar2imu[0], r_lidar2imu[1], r_lidar2imu[2],
           r_lidar2imu[3], r_lidar2imu[4], r_lidar2imu[5],
           r_lidar2imu[6], r_lidar2imu[7], r_lidar2imu[8];

    // 计算Lidar到IMU的逆变换（即IMU到Lidar的变换）
    // Rli_: Lidar坐标系到IMU坐标系的旋转矩阵 = Ril的转置，形状为 (3, 3)
    Rli_ = Ril.transpose();
    // tli_: Lidar坐标系到IMU坐标系的平移向量，形状为 (3,)
    // 公式: t_li = -R_li * t_il
    tli_ = -Rli_ * til;

    // 计算Camera到IMU的外参变换
    // Rci_: Camera坐标系到IMU坐标系的旋转矩阵，形状为 (3, 3)
    // 公式: R_ci = R_cl * R_li（链式变换：Camera->Lidar->IMU）
    Rci_ = Rcl_ * Rli_;
    // tci_: Camera坐标系到IMU坐标系的平移向量，形状为 (3,)
    // 公式: t_ci = R_cl * t_li + t_cl
    tci_ = Rcl_ * tli_ + tcl_;
}


// 不需要包含 <GL/gl.h>，避免 GL_LUMINANCE / GL_UNSIGNED_BYTE 冲突
// 但要确保项目已链接 DevIL、GLEW/GLUT（SiftGPU 依赖）
bool LvbaSystem::loadFromColmapDB()
{
    constexpr uint64_t kColmapMaxNumImages = (1ull << 31) - 1;
    auto imageIdsToPairId = [kColmapMaxNumImages](uint32_t image_id1, uint32_t image_id2) -> uint64_t {
        if (image_id1 > image_id2) {
            std::swap(image_id1, image_id2);
        }
        return static_cast<uint64_t>(image_id1) * kColmapMaxNumImages +
               static_cast<uint64_t>(image_id2);
    };

    sqlite3* db = nullptr;
    if (sqlite3_open(dataset_io_->colmap_db_path_.c_str(), &db) != SQLITE_OK) {
        std::cerr << "[DB] open failed: " << sqlite3_errmsg(db) << "\n";
        return false;
    }

    // 1) 读取 images 表并将 image_id 和 file_name 对应起来
    std::unordered_map<std::string, uint32_t> name2id;
    size_t db_image_count = 0;
    {
        const char* sql = "SELECT image_id, name FROM images;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) == SQLITE_OK) {
            while (sqlite3_step(stmt) == SQLITE_ROW) {
                uint32_t id = (uint32_t)sqlite3_column_int64(stmt, 0);
                std::string name = reinterpret_cast<const char*>(sqlite3_column_text(stmt, 1));
                name2id[name] = id;
                name2id[fs::path(name).filename().string()] = id;
                ++db_image_count;
            }
        }
        sqlite3_finalize(stmt);
    }
    std::cout << "[DB] ColmapDB images count = " << db_image_count << "\n";

    if (db_image_count != images_ids_.size()) {
        std::cerr << "[DB] Warning: DB images count (" << db_image_count
                  << ") != dataset images count (" << images_ids_.size() << ")\n";
        // 构建新的 数据库
        std::cout << "[DB] Rebuilding COLMAP database...\n";
        sqlite3_close(db);
        return false;
    }

    ts2idx.clear();
    ts2idx.reserve(images_ids_.size());
    for (int i = 0; i < (int)images_ids_.size(); ++i) {
        ts2idx[images_ids_[i]] = i;
    }

    // 辅助函数：根据时间戳查找对应的图像 ID
    auto imageIdOfTs = [&](double ts)->int {
        std::string p = getImagePath(ts);
        std::string base = fs::path(p).filename().string();
        auto it1 = name2id.find(base);
        if (it1 != name2id.end()) return (int)it1->second;
        return -1;
    };

    // 2) 读取 keypoints 表并写入 all_keypoints_
    if ((int)all_keypoints_.size() < (int)images_ids_.size())
        all_keypoints_.resize(images_ids_.size());

    {
        const char* sql = "SELECT rows, cols, data FROM keypoints WHERE image_id=?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            std::cerr << "[DB] keypoints SELECT prepare failed.\n";
        } else {
            for (size_t i = 0; i < images_ids_.size(); ++i) {
                int image_id = imageIdOfTs(images_ids_[i]);
                if (image_id < 0) continue;

                sqlite3_reset(stmt);
                sqlite3_bind_int(stmt, 1, image_id);
                if (sqlite3_step(stmt) == SQLITE_ROW) {
                    int rows = sqlite3_column_int(stmt, 0);
                    int cols = sqlite3_column_int(stmt, 1);  // 4 或 6
                    const void* blob = sqlite3_column_blob(stmt, 2);
                    int blob_bytes = sqlite3_column_bytes(stmt, 2);

                    if (blob && blob_bytes == rows * cols * (int)sizeof(float)) {
                        const float* fp = reinterpret_cast<const float*>(blob);
                        auto& vec = all_keypoints_[i];
                        vec.resize(rows);
                        for (int r = 0; r < rows; ++r) {
                            sift::Keypoint kp{};
                            kp.x = fp[r * cols + 0];
                            kp.y = fp[r * cols + 1];
                            if (cols >= 3) kp.sigma = fp[r * cols + 2];
                            if (cols >= 4) kp.extremum_val = fp[r * cols + 3];
                            vec[r] = kp;
                        }
                    }
                }
            }
        }
        sqlite3_finalize(stmt);
    }

    // 3) 读取 two_view_geometries 表并写入 all_matches_（只读取内点匹配）
    all_matches_.assign(image_pairs_.size(), {});
    {
        const char* sql = "SELECT rows, cols, data FROM two_view_geometries WHERE pair_id=?;";
        sqlite3_stmt* stmt = nullptr;
        if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
            std::cerr << "[DB] two_view_geometries SELECT prepare failed.\n";
        } else {
            for (size_t k = 0; k < image_pairs_.size(); ++k) {
                const double ts1 = image_pairs_[k].first;
                const double ts2 = image_pairs_[k].second;

                const int id1 = imageIdOfTs(ts1);   // DB 的 image_id
                const int id2 = imageIdOfTs(ts2);
                if (id1 < 0 || id2 < 0) continue;

                // 关键点容器索引
                int idx1 = ts2idx[ts1]; // images_ids_ 的顺序下标
                int idx2 = ts2idx[ts2];
                if (idx1 < 0 || idx1 >= (int)all_keypoints_.size() ||
                    idx2 < 0 || idx2 >= (int)all_keypoints_.size()) {
                    continue;
                }
                const auto& kpts1 = all_keypoints_[idx1];
                const auto& kpts2 = all_keypoints_[idx2];
                if (kpts1.empty() || kpts2.empty()) {
                    std::cout << "[DB] keypoints of (" << ts1 << "," << ts2 << ") is empty.\n";
                    continue;
                }

                // 你的 pair_id 计算已保证 image_id 升序
                const bool swapped = (id1 > id2);  // 仅用于索引方向校正
                const uint64_t pair_id = imageIdsToPairId((uint32_t)id1, (uint32_t)id2);

                sqlite3_reset(stmt);
                sqlite3_bind_int64(stmt, 1, (sqlite3_int64)pair_id);
                if (sqlite3_step(stmt) != SQLITE_ROW) continue;

                const int rows = sqlite3_column_int(stmt, 0);
                const int cols = sqlite3_column_int(stmt, 1); // 应为 2
                const void* blob = sqlite3_column_blob(stmt, 2);
                const int blob_bytes = sqlite3_column_bytes(stmt, 2);
                if (cols != 2 || !blob || rows <= 0 ||
                    blob_bytes != rows * 2 * (int)sizeof(uint32_t)) {
                    continue;
                }

                const uint32_t* up = reinterpret_cast<const uint32_t*>(blob);
                auto& vec = all_matches_[k];
                vec.reserve(rows);

                for (int r = 0; r < rows; ++r) {
                    int i_small = (int)up[2*r + 0];  // 索引对应 "较小 image_id" 那一侧
                    int i_large = (int)up[2*r + 1];  // 索引对应 "较大 image_id" 那一侧

                    // 若当前(ts1,ts2)的 id 顺序与 pair_id 的顺序不一致，则交换
                    int i1 = i_small;
                    int i2 = i_large;
                    if (swapped) std::swap(i1, i2);

                    // 防御性过滤，避免 drawMatches 越界
                    if (i1 >= 0 && i1 < (int)kpts1.size() &&
                        i2 >= 0 && i2 < (int)kpts2.size()) {
                        vec.emplace_back(i1, i2);
                    }
                }
            }
        }
        sqlite3_finalize(stmt);
    }

    sqlite3_close(db);
    std::cout << "[DB] Loaded keypoints & inlier matches from " << dataset_io_->colmap_db_path_ << "\n";
    // 记录结束时间
    return true;
}

/**
 * [功能描述]：使用SiftGPU进行GPU加速的SIFT特征提取和匹配。
 *            对所有图像对进行特征提取、描述子匹配，并保存匹配结果和可视化图像。
 *            支持从COLMAP数据库加载已有结果以避免重复计算。
 * @return 无返回值，结果存储在all_keypoints_和all_matches_成员变量中
 */
void LvbaSystem::extractAndMatchFeaturesGPU()
{
    // ========== 第一步：建立时间戳到顺序索引的映射 ==========
    // ts2idx: 时间戳 -> 顺序下标映射，用于保持all_keypoints_的按序存储
    std::unordered_map<double, int> ts2idx;
    ts2idx.reserve(images_ids_.size());
    for (int i = 0; i < (int)images_ids_.size(); ++i) ts2idx[images_ids_[i]] = i;

    // 确保关键点容器大小足够
    if ((int)all_keypoints_.size() < (int)images_ids_.size())
        all_keypoints_.resize(images_ids_.size());

    // ========== 第二步：尝试从COLMAP数据库加载已有结果 ==========
    if (loadFromColmapDB()) {
        std::cout << "[Frontend] Using existing COLMAP DB: "
                  << dataset_io_->colmap_db_path_ << "\n";
        return;  // 已经拿到结果，直接返回
    }

    // ========== 第三步：初始化SiftGPU特征提取器 ==========
    SiftGPU sift;
    // SiftGPU参数说明：
    // -fo -1: 第一个八度从原图开始
    // -loweo: 低对比度边缘点过滤
    // -w 3: 描述子窗口大小
    // -t 0.01: 对比度阈值
    // -e 12: 边缘阈值
    // -v 0: 静默模式
    const char* argv[] = {"-fo","-1","-loweo","-w","3","-t","0.01","-e","12","-v","0"};
    sift.ParseParam(10, const_cast<char**>(argv));
    
    // 创建OpenGL上下文
    if (sift.CreateContextGL() != SiftGPU::SIFTGPU_FULL_SUPPORTED) {
        std::cerr << "[SiftGPU] Not supported or context creation failed.\n";
        return;
    }

    // ========== 第四步：初始化SiftGPU特征匹配器 ==========
    SiftMatchGPU matcher;
    matcher.VerifyContextGL();
    // const float kRatioMax  = 1.0f;  // Lowe 比值阈值（越小越严格）0.85
    // const float kDistMax   = 0.5f;  // 距离上限（按 SiftGPU 自己的度量，先给个中性值）
    // const int   kUseMBM    = 1;      // mutual-best-match（双向最邻近）
    // matcher.SetRatio(0.8f);

    // ========== 第五步：定义图像缓存结构和特征提取函数 ==========
    // 缓存结构：按时间戳存储已提取的特征，避免重复提取
    struct ImgCache {
        std::vector<float> desc;                  // SIFT描述子，维度为 128*N（N为特征点数量）
        std::vector<SiftGPU::SiftKeypoint> kpt;   // SiftGPU格式的关键点
        std::vector<sift::Keypoint> kpt_conv;     // 转换后的Keypoint格式（包含x,y坐标）
    };
    std::unordered_map<double, ImgCache> cache;
    cache.reserve(image_pairs_.size() * 2);

    // ---------- 特征提取Lambda函数 ----------
    // 功能：对给定时间戳的图像提取SIFT特征，结果缓存到cache中
    // @param ts: 图像时间戳
    // @return: 提取成功返回true，失败返回false
    auto extract_once = [&](double ts) -> bool {
        // 如果已缓存，直接返回
        if (cache.find(ts) != cache.end()) return true;

        // 读取图像
        std::string path = getImagePath(ts);
        cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);
        // cv::Mat img = preprocessLowTextureBGR(img0, false);

        // 如果图像尺寸不匹配，进行缩放
        if (img.cols != image_width_ || img.rows != image_height_) 
        {
            cv::resize(img, img, cv::Size(img.cols * scale_, img.rows * scale_), 0, 0, cv::INTER_LINEAR);
        }

        if (img.empty()) {
            std::cerr << "[SiftGPU] imread failed: " << path << "\n";
            return false;
        }

        // 运行SIFT特征提取（OpenCV图像格式为BGR）
        if (!sift.RunSIFT(img.cols, img.rows, img.data, GL_BGR, GL_UNSIGNED_BYTE)) {
            std::cerr << "[SiftGPU] RunSIFT failed for " << path << "\n";
            return false;
        }

        // 获取特征数量并分配内存
        int n = sift.GetFeatureNum();
        ImgCache entry;
        entry.desc.resize(128 * n);  // 每个SIFT描述子128维
        entry.kpt.resize(n);
        // 获取特征向量（关键点和描述子）
        if (n > 0) sift.GetFeatureVector(entry.kpt.data(), entry.desc.data());

        // 将SiftGPU关键点格式转换为自定义Keypoint格式
        entry.kpt_conv.resize(n);
        for (int i = 0; i < n; ++i) {
            sift::Keypoint kp{};
            kp.x = entry.kpt[i].x;  // 特征点x坐标
            kp.y = entry.kpt[i].y;  // 特征点y坐标
            // 如需：kp.scale = entry.kpt[i].s; kp.orientation = entry.kpt[i].o;
            entry.kpt_conv[i] = kp;
        }

        // 将提取的关键点写回all_keypoints_（按顺序下标存储）
        auto it = ts2idx.find(ts);
        if (it != ts2idx.end()) {
            int idx = it->second;
            if (all_keypoints_[idx].empty())
                all_keypoints_[idx] = entry.kpt_conv;
        }

        // 将结果加入缓存
        cache.emplace(ts, std::move(entry));
        return true;
    };

    // ========== 第六步：遍历图像对进行特征匹配 ==========
    all_matches_.clear();
    all_matches_.resize(image_pairs_.size());
    std::vector<bool> is_bad_pair(image_pairs_.size(), false);
    int pair_count = 0;
    
    std::cout << "[SiftGPU] Start Matching " << image_pairs_.size() << " image pairs ...\n";
    for (const auto& pr : image_pairs_) {
        double ts1 = pr.first, ts2 = pr.second;  // 图像对的两个时间戳
        
        // ---------- 6.1 提取两张图像的特征 ----------
        if (!extract_once(ts1) || !extract_once(ts2)) {
            std::cout << "[SiftGPU] extract_once failed for pair: "
                      << ts1 << " and " << ts2 << "\n";
            all_matches_[pair_count].clear();
            continue;
        }

        // ---------- 6.2 设置匹配器的描述子 ----------
        auto& c1 = cache[ts1];  // 第一张图像的缓存
        auto& c2 = cache[ts2];  // 第二张图像的缓存
        matcher.SetDescriptors(0, (int)c1.kpt.size(), c1.desc.data());  // 设置图像1的描述子
        matcher.SetDescriptors(1, (int)c2.kpt.size(), c2.desc.data());  // 设置图像2的描述子

        // ---------- 6.3 执行特征匹配 ----------
        std::vector<std::pair<int,int>> matches;  // 匹配结果：(图像1特征索引, 图像2特征索引)
        if (!c1.kpt.empty()) {
            // 分配匹配结果缓冲区
            std::unique_ptr<int[][2]> buf(new int[c1.kpt.size()][2]);
            // GetSiftMatch参数：最大匹配数、输出缓冲区、距离阈值、比值阈值、是否双向匹配
            int nmatch = matcher.GetSiftMatch((int)c1.kpt.size(), buf.get(), 0.7f, 0.8f, 1);
            matches.reserve(nmatch);
            
            // 验证匹配索引有效性并存储
            for (int i = 0; i < nmatch; ++i) {
                int i1 = buf[i][0], i2 = buf[i][1];  // 匹配的特征点索引对
                if (i1 >= 0 && i1 < (int)c1.kpt.size() &&
                    i2 >= 0 && i2 < (int)c2.kpt.size()) {
                    matches.emplace_back(i1, i2);
                }
            }
        }
        // std::cout << "matches.size(): " << matches.size() << std::endl;
        
        // 保存当前图像对的匹配结果
        all_matches_[pair_count] = matches;

        // ---------- 6.4 可视化匹配结果 ----------
        // 使用ts2idx映射得到顺序下标，用于文件命名
        int id1 = ts2idx[ts1];
        int id2 = ts2idx[ts2];
        cv::Mat img1 = cv::imread(getImagePath(ts1), cv::IMREAD_COLOR);
        cv::Mat img2 = cv::imread(getImagePath(ts2), cv::IMREAD_COLOR);
        // 绘制并保存匹配可视化图像
        drawAndSaveMatchesGPU(dataset_path_ + "result/", id1, id2, img1, img2, c1.kpt, c2.kpt, matches);
        pair_count++;
        // if (pair_count == 20) {
        //     std::cout << std::endl;
        //     std::cin.get(); // 仅示例，暂停查看前10个结果
        // }

        printProgressBar(pair_count, image_pairs_.size());
    }
    std::cout << std::endl;
}

/**
 * [功能描述]：利用体素化的点云为每张图像生成深度图。
 *            通过将网格地图中的3D点投影到图像平面，考虑相机畸变模型，
 *            生成每个像素的深度值（取最近深度），并将深度图保存为PNG文件。
 * @return 无返回值，结果存储在all_depths_、Rcw_all_、tcw_all_等成员变量中
 */
void LvbaSystem::generateDepthWithVoxel() 
{
    // ========== 第一步：数据一致性检查 ==========
    const size_t N = all_voxel_ids_.size();  // 图像数量
    if (poses_.size() != N) {
        std::cerr << "[generateDepthWithVoxel] size mismatch: poses=" << poses_.size()
                  << ", voxel_ids=" << N << std::endl;
    }
    if (!images_ids_.empty() && images_ids_.size() != N) {
        std::cerr << "[generateDepthWithVoxel] size mismatch: images_ids=" << images_ids_.size()
                  << ", voxel_ids=" << N << std::endl;
    }

    // std::cout << "all_voxel_ids_.size(): " << N << std::endl;

    // ========== 第二步：清空并预分配输出容器 ==========
    Rcw_all_.clear(); tcw_all_.clear(); Rcw_all_optimized_.clear(); tcw_all_optimized_.clear(); all_depths_.clear();
    Rcw_all_.reserve(N); tcw_all_.reserve(N); Rcw_all_optimized_.reserve(N); tcw_all_optimized_.reserve(N); all_depths_.reserve(N);
    
    // ========== 第三步：遍历每张图像生成深度图 ==========
    std::cout << "[generateDepthWithVoxel] Generating depths for " << N << " images ...\n";
    for (size_t id = 0; id < N; ++id) 
    {
        // ---------- 3.1 计算优化后的相机外参（世界到相机的变换） ----------
        // T_W_I_opt: 优化后的IMU在世界坐标系下的位姿
        const Sophus::SE3& T_W_I_opt = poses_[id];
        const Eigen::Matrix3d Rwi_opt = T_W_I_opt.rotation_matrix();  // IMU到世界的旋转，形状为 (3, 3)
        const Eigen::Vector3d Pwi_opt = T_W_I_opt.translation();      // IMU在世界系下的位置，形状为 (3,)

        // 计算世界到相机的变换：T_cw = T_ci * T_iw = T_ci * T_wi^(-1)
        // Rcw_: 世界到相机的旋转矩阵，形状为 (3, 3)
        // 公式: Rcw = Rci * Rwi^T
        Rcw_ = Rci_ * Rwi_opt.transpose();
        // tcw_: 世界到相机的平移向量，形状为 (3,)
        // 公式: tcw = -Rcw * Pwi + tci
        tcw_ = -Rcw_ * Pwi_opt + tci_;
        Rcw_all_optimized_.push_back(Rcw_);
        tcw_all_optimized_.push_back(tcw_);

        // ---------- 3.2 计算原始（未优化）的相机外参 ----------
        const Sophus::SE3& T_W_I_orig = poses_before_[id];
        const Eigen::Matrix3d Rwi_orig = T_W_I_orig.rotation_matrix();
        const Eigen::Vector3d Pwi_orig = T_W_I_orig.translation();
        Eigen::Matrix3d Rcw_orig = Rci_ * Rwi_orig.transpose();
        Eigen::Vector3d tcw_orig = -Rcw_orig * Pwi_orig + tci_;
        Rcw_all_.push_back(Rcw_orig);
        tcw_all_.push_back(tcw_orig);

        // ---------- 3.3 初始化深度图 ----------
        // 创建单通道32位浮点深度图，初始值为0
        cv::Mat depth(image_height_, image_width_, CV_32FC1, cv::Scalar(0));

        // 获取当前图像对应的体素ID列表
        const auto& voxel_ids = all_voxel_ids_[id];

        // ---------- 3.4 遍历体素，将点投影到图像生成深度 ----------
        for (const VOXEL_LOC& voxel_xyz : voxel_ids) 
        {
            // 在网格地图中查找该体素
            VOXEL_LOC position(voxel_xyz.x, voxel_xyz.y, voxel_xyz.z);
            auto it = grid_map_.find(position);
            if (it == grid_map_.end()) continue;  // 体素不存在，跳过

            // 获取该体素内的所有3D点
            const std::vector<Eigen::Vector3d>& points = it->second;

            // 遍历体素内的每个点
            for (const auto& pW : points)
            {
                // 将世界坐标系的点变换到相机坐标系
                // pC = Rcw * pW + tcw，形状为 (3,)
                Eigen::Vector3d pC = Rcw_ * pW + tcw_;
                
                // 获取深度值（相机Z轴方向的距离）
                const double Z = pC.z();
                if (Z < 1e-3) continue;  // 深度太小（在相机后方或太近），跳过

                // 归一化平面坐标
                const double x = pC.x() / Z;
                const double y = pC.y() / Z;

                // 应用畸变模型（Brown-Conrady畸变模型）
                // r2: 径向距离的平方
                const double r2 = x * x + y * y;
                // x_dist, y_dist: 畸变后的归一化坐标
                // 径向畸变: (1 + k1*r2 + k2*r4)
                // 切向畸变: 2*p1*x*y + p2*(r2 + 2*x^2) 和 p1*(r2 + 2*y^2) + 2*p2*x*y
                const double x_dist = x * (1 + d0_ * r2 + d1_ * r2 * r2)
                                    + 2 * d2_ * x * y + d3_ * (r2 + 2 * x * x);
                const double y_dist = y * (1 + d0_ * r2 + d1_ * r2 * r2)
                                    + d2_ * (r2 + 2 * y * y) + 2 * d3_ * x * y;

                // 投影到像素坐标
                // u = fx * x_dist + cx, v = fy * y_dist + cy
                const int u = static_cast<int>(fx_ * x_dist + cx_);
                const int v = static_cast<int>(fy_ * y_dist + cy_);
                
                // 边界检查
                if (u < 0 || u >= image_width_ || v < 0 || v >= image_height_) continue;

                // 更新深度图（保留最近的深度值，即Z-buffer算法）
                float& d = depth.at<float>(v, u);
                if (d == 0.f || Z < d) d = static_cast<float>(Z);
            }
        }

        // ---------- 3.5 保存深度图 ----------
        all_depths_.push_back(depth);
        printProgressBar(all_depths_.size(), all_voxel_ids_.size());
        
        // 保存深度图为PNG文件，以时间戳命名
        if (!images_ids_.empty()) {
            std::ostringstream oss;
            oss.setf(std::ios::fixed); oss << std::setprecision(6) << images_ids_[id];
            const std::string out = dataset_path_ + "depth/" + oss.str() + ".png";
            cv::Mat vis;
            // 转换为16位无符号整数，缩放因子2000（即1米对应2000）
            depth.convertTo(vis, CV_16UC1, 2000.0); // 1m -> 1000
            cv::imwrite(out, vis);
        }
    }
    std::cout << std::endl;

}

/**
 * [功能描述]：从图像特征匹配结果构建特征轨迹（Feature Tracks）并融合生成3D地图点。
 *            通过BFS遍历匹配图的连通分量，对每个连通分量进行深度反投影、距离筛选、
 *            去重、重投影误差检验和视角过滤，最终生成高质量的3D地图点。
 * @return 无返回值，结果存储在tracks_成员变量中
 */
void LvbaSystem::BuildTracksAndFuse3D() {

    const int N = static_cast<int>(all_keypoints_.size());  // 图像数量
    std::cout << "[BuildTracksAndFuse3D] Building visual points from " << N << " images ...\n";
    
    // ========== 第一步：初始化数据结构 ==========
    // obs_to_track: 记录每个观测点(image_id, keypoint_id)属于哪个track
    // -1表示未处理，-2表示正在BFS中，>=0表示属于某个track
    std::vector<std::vector<int>> obs_to_track(N);
    for (int i = 0; i < N; ++i) {
        obs_to_track[i].assign((int)all_keypoints_[i].size(), -1);
    }

    // ========== 第二步：构建特征匹配的邻接表 ==========
    // adj[i][ki]: 图像i的第ki个特征点在其他图像中的匹配点列表
    // 格式: vector<pair<图像索引j, 特征点索引kj>>
    std::vector<std::vector<std::vector<std::pair<int,int>>>> adj(N);
    for (int i = 0; i < N; ++i) adj[i].resize(all_keypoints_[i].size());

    // 从all_matches_构建双向邻接关系
    for (int i = 0; i < N-1; ++i) {
        for (int j = i+1; j < N; ++j) {
            // 获取图像对(i,j)的匹配结果索引
            size_t idx = pairIndex(i, j, N);
            const auto& matches_ij = all_matches_[idx];
            if (matches_ij.empty()) continue;
            
            // 遍历匹配对，建立双向邻接关系
            for (const auto& m : matches_ij) {
                int ki = m.first;   // 图像i中的特征点索引
                int kj = m.second;  // 图像j中的特征点索引
                if (ki < 0 || kj < 0) continue;
                if (ki >= (int)all_keypoints_[i].size() || 
                    kj >= (int)all_keypoints_[j].size()) continue;
                adj[i][ki].push_back({j, kj});  // i->j
                adj[j][kj].push_back({i, ki});  // j->i
            }
        }
    }

    // 清空并预分配轨迹容器
    tracks_.clear();
    tracks_.reserve(100000);

    int num_tracked = 0;        // 成功构建的track数量
    size_t total_components = 0; // 总连通分量数量
    
    // ========== 第三步：BFS遍历连通分量，构建轨迹 ==========
    for (int i = 0; i < N; ++i) {
        for (int ki = 0; ki < (int)all_keypoints_[i].size(); ++ki) {
            // 跳过已处理的观测点
            if (obs_to_track[i][ki] != -1) continue;

            // ---------- 3.1 BFS搜索连通分量 ----------
            // component: 当前连通分量中的所有观测点 (image_id, keypoint_id)
            std::vector<std::pair<int,int>> component;
            std::deque<std::pair<int,int>> q;  // BFS队列
            q.push_back({i, ki});
            obs_to_track[i][ki] = -2;  // 标记为正在处理

            // BFS遍历
            while (!q.empty()) {
                auto cur = q.front(); q.pop_front();
                component.push_back(cur);
                int ci = cur.first;   // 当前图像索引
                int ck = cur.second;  // 当前特征点索引
                
                // 遍历所有邻居（匹配的特征点）
                for (auto& nb : adj[ci][ck]) {
                    int ni = nb.first, nk = nb.second;
                    if (obs_to_track[ni][nk] == -1) {
                        obs_to_track[ni][nk] = -2;  // 标记为正在处理
                        q.push_back(nb);
                    }
                }
            }
            ++total_components;

            // ---------- 3.2 观测数量阈值检查 ----------
            // 连通分量太小则跳过
            if ((int)component.size() < obser_thr_) {
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            // ---------- 3.3 反投影所有观测点得到3D坐标 ----------
            // points3d: 每个观测点反投影得到的世界坐标系3D点
            std::vector<Eigen::Vector3d> points3d(component.size(), Eigen::Vector3d::Zero());
            std::vector<int> valid_mask(component.size(), 0);  // 有效点标记
            
            for (size_t t = 0; t < component.size(); ++t) {
                int im = component[t].first;   // 图像索引
                int kp = component[t].second;  // 特征点索引
                float u = all_keypoints_[im][kp].x;  // 像素坐标u
                float v = all_keypoints_[im][kp].y;  // 像素坐标v

                // 双线性插值获取深度值
                float d = -1.0f;
                if (!fetchDepthBilinear(all_depths_[im], u, v, d, 0.001f)) continue;
                if (d <= 0.0f) continue;

                // 反投影到相机坐标系，再变换到世界坐标系
                Eigen::Vector3d Xc = backProjectCam(u, v, d, fx_, fy_, cx_, cy_);  // 相机系3D点
                Eigen::Vector3d Xw = camToWorld(Xc, Rcw_all_optimized_[im], tcw_all_optimized_[im]);  // 世界系3D点
                points3d[t] = Xw;
                valid_mask[t] = 1;
            }

            // 收集有效点的索引
            std::vector<int> idx_valid;
            for (size_t t = 0; t < points3d.size(); ++t) if (valid_mask[t]) idx_valid.push_back((int)t);
            if ((int)idx_valid.size() < obser_thr_) {
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            // ---------- 3.4 基于距离的Inlier筛选 ----------
            // 以第一个有效点为锚点，筛选距离在阈值内的点
            Eigen::Vector3d anchor = points3d[idx_valid[0]];
            
            std::vector<int> inliers;
            inliers.reserve(idx_valid.size());
            for (int id : idx_valid) {
                double dist = (points3d[id] - anchor).norm();
                if (dist < 0.12) inliers.push_back(id);  // 距离阈值0.12米
            }
            if ((int)inliers.size() < obser_thr_) {
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            // ---------- 3.5 按图像ID去重（每图只保留一个观测） ----------
            // best_id: 图像ID -> 选中的component索引
            std::unordered_map<int,int> best_id;
            best_id.reserve(inliers.size());
            for (int id : inliers) {
                int img_id = component[id].first;
                if (!best_id.count(img_id)) best_id[img_id] = id;  // 每图只取第一个
            }
            if ((int)best_id.size() < obser_thr_) {
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            // ---------- 3.6 融合3D点（计算均值） ----------
            Eigen::Vector3d Xw_fused = Eigen::Vector3d::Zero();
            for (const auto& kv : best_id)
            {
                const int comp_idx = kv.second;
                Xw_fused += points3d[comp_idx];
            }
            Xw_fused /= double(best_id.size());  // 取平均作为融合点

            if (Xw_fused.norm() < 0.1) {
                std::cout << "bad fused point: " << Xw_fused.transpose() << std::endl;
            }

            // ---------- 3.7 重投影误差检验 ----------
            // 计算融合点到各观测的平均重投影误差
            double sum_err = 0.0;
            int cnt_err = 0;

            for (const auto& kv : best_id) {
                const int comp_idx = kv.second;
                const int img_id = component[comp_idx].first;
                const int kp_id  = component[comp_idx].second;

                // 边界检查
                if (img_id < 0 || img_id >= (int)Rcw_all_optimized_.size() || img_id >= (int)tcw_all_optimized_.size()) continue;
                if (img_id < 0 || img_id >= (int)all_keypoints_.size() || kp_id < 0 || kp_id >= (int)all_keypoints_[img_id].size()) continue;

                // 观测的像素坐标
                const double u_obs = all_keypoints_[img_id][kp_id].x;
                const double v_obs = all_keypoints_[img_id][kp_id].y;

                // 将融合点投影到当前图像
                const Eigen::Matrix3d& Rcw = Rcw_all_optimized_[img_id];
                const Eigen::Vector3d& tcw = tcw_all_optimized_[img_id];
                const Eigen::Vector3d Xc = Rcw * Xw_fused + tcw;  // 世界系->相机系
                if (Xc.z() <= 1e-9) continue;

                // 相机系->像素坐标（针孔模型投影）
                const double invz = 1.0 / Xc.z();
                const double u_hat = fx_ * (Xc.x() * invz) + cx_;
                const double v_hat = fy_ * (Xc.y() * invz) + cy_;

                // 计算重投影误差（欧氏距离）
                const double du = u_hat - u_obs;
                const double dv = v_hat - v_obs;
                const double err = std::sqrt(du * du + dv * dv);

                sum_err += err;
                cnt_err++;
            }

            // 有效观测数不足则跳过
            if (cnt_err < obser_thr_) {
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            // 平均重投影误差超过阈值则剔除
            const double mean_reproj = sum_err / double(cnt_err);
            if (mean_reproj > reproj_mean_thr_px_) {

                std::cout << "[TrackFilter] drop by mean reproj=" << mean_reproj
                          << " thr=" << reproj_mean_thr_px_ << " cnt=" << cnt_err << std::endl;

                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }

            // ---------- 3.8 视角多样性过滤 ----------
            // 确保观测来自足够不同的视角，避免退化情况
            const double cos_min_view_angle = std::cos(min_view_angle_deg_ * M_PI / 180.0);

            std::vector<int> kept_obs_ids;        // 保留的观测索引
            std::vector<Eigen::Vector3d> kept_dirs;  // 保留观测的视线方向
            kept_obs_ids.reserve(best_id.size());
            kept_dirs.reserve(best_id.size());

            for (auto &kv : best_id) {
                const int comp_idx = kv.second;
                const int cam_id = kv.first;
                if (cam_id < 0 || cam_id >= (int)Rcw_all_optimized_.size() ||
                    cam_id >= (int)tcw_all_optimized_.size()) {
                    continue;
                }
                
                // 计算相机光心在世界坐标系中的位置
                // Cw = -Rcw^T * tcw
                const Eigen::Matrix3d& Rcw = Rcw_all_optimized_[cam_id];
                const Eigen::Vector3d& tcw = tcw_all_optimized_[cam_id];
                const Eigen::Vector3d Cw = -Rcw.transpose() * tcw;

                // 计算从相机光心到3D点的视线方向（归一化）
                Eigen::Vector3d dir = points3d[comp_idx] - Cw;
                const double dir_norm = dir.norm();
                if (dir_norm < 1e-6) continue;
                dir /= dir_norm;

                // 检查与已保留视线方向的最小夹角
                double min_dot = 1.0;  // cos(0°) = 1
                for (const auto& d : kept_dirs) {
                    const double dot = dir.dot(d);  // 点积 = cos(夹角)
                    if (dot < min_dot) min_dot = dot;
                }
                
                // 如果夹角足够大（cos值足够小），则保留该观测
                if (kept_dirs.empty() || min_dot <= cos_min_view_angle) {
                    kept_obs_ids.push_back(comp_idx);
                    kept_dirs.push_back(dir);
                }
            }

            // auto log_track_stats = [&](const char* tag) {
            //     static int dbg_cnt = 0;
            //     if (dbg_cnt < 10 || (dbg_cnt % 200 == 0)) {
            //         std::cout << "[TrackFilter] " << tag
            //                   << " comp=" << component.size()
            //                   << " inliers=" << inliers.size()
            //                   << " best=" << best_id.size()
            //                   << " kept=" << kept_obs_ids.size()
            //                   << std::endl;
            //     }
            //     ++dbg_cnt;
            // };

            // 视角过滤后观测数不足则跳过
            if (kept_obs_ids.empty() || (int)kept_obs_ids.size() < obser_thr_) {
                // log_track_stats("drop");
                for (auto &obs : component) obs_to_track[obs.first][obs.second] = -1;
                continue;
            }
            // log_track_stats("keep");

            // Eigen::Vector3d Xw_fused = Eigen::Vector3d::Zero();
            // for (int id : kept_obs_ids) Xw_fused += points3d[id];
            // Xw_fused /= double(kept_obs_ids.size());

            // if (Xw_fused.norm() < 0.1) {
            //     std::cout << "bad fused point: " << Xw_fused.transpose() << std::endl;
            // }

            // RMS / MAD 基于去重后的集合
            // double sqsum = 0.0;
            // std::vector<double> resid_in;
            // resid_in.reserve(kept_obs_ids.size());
            // for (int id : kept_obs_ids) {
            //     double r = (points3d[id] - Xw_fused).norm();
            //     resid_in.push_back(r);
            //     sqsum += r * r;
            // }
            // double rms = std::sqrt(sqsum / double(resid_in.size()));
            // double mad_in = computeMAD(resid_in);

            // ---------- 3.9 创建Track对象并保存 ----------
            Track tr;
            tr.Xw_fused = Xw_fused;           // 融合后的3D点坐标
            // tr.mad = mad_in;
            // tr.rms = rms;
            tr.observations = component;      // 所有观测点
            tr.inlier_indices.reserve(kept_obs_ids.size());
            for (int id : kept_obs_ids) tr.inlier_indices.push_back(id);  // 保留的inlier索引

            // 将Track添加到列表，并更新obs_to_track映射
            int track_id = (int)tracks_.size();
            tracks_.push_back(std::move(tr));
            ++num_tracked;

            // 将该连通分量的所有观测点标记为属于当前track
            for (auto &obs : component) obs_to_track[obs.first][obs.second] = track_id;
        }
    }
    
    // ========== 第四步：输出统计信息 ==========
    if (total_components > 0) {
        const size_t kept = static_cast<size_t>(num_tracked);
        const size_t dropped = (total_components >= kept) ? (total_components - kept) : 0;
        const double keep_ratio = 100.0 * static_cast<double>(kept) / static_cast<double>(total_components);
        std::cout << "[TrackFilter] kept=" << kept << " dropped=" << dropped
                  << " total=" << total_components
                  << " ratio=" << std::fixed << std::setprecision(2)
                  << keep_ratio << "%" << std::defaultfloat << std::endl;
    }
    
    // 备份优化前的tracks
    tracks_before_ = tracks_;

    // showTracksComparePCL();
    // saveTrackFeaturesOnImages();
}


/**
 * [功能描述]：从优化后的LiDAR位姿和点云数据构建全局网格地图（Grid Map）。
 *            该函数将所有帧的点云变换到世界坐标系，按体素存储到grid_map_中，
 *            并为每张图像建立时间窗口内的体素索引列表，用于后续深度图生成。
 * @return 无返回值，结果存储在成员变量grid_map_和all_voxel_ids_中
 */
void LvbaSystem::buildGridMapFromOptimized() {
    // 清空全局网格地图
    grid_map_.clear();
    
    // 获取优化后的位姿和点云数据
    const auto& x_buf_full = dataset_io_->x_buf_;       // 所有帧的位姿
    const auto& pl_fulls_full = dataset_io_->pl_fulls_; // 所有帧的点云

    // 检查数据有效性
    const size_t N = std::min(x_buf_full.size(), pl_fulls_full.size());
    if (N == 0) {
        ROS_WARN("buildGridMapFromOptimized: empty inputs, skip.");
        return;
    }

    const double vox = 0.5;  // 体素大小（单位：米）

    // ========== 第一步：将所有点云变换到世界坐标系并进行体素化 ==========
    size_t total_points = 0;  // 总点数统计
    // per_frame_voxels: 记录每帧点云占据的体素集合，用于后续时间窗口查询
    std::vector<std::set<VOXEL_LOC>> per_frame_voxels(N);
    
    for (size_t i = 0; i < N; ++i) {
        // 获取当前帧的旋转矩阵R和平移向量t
        const Eigen::Matrix3d& R = x_buf_full[i].R;  // 旋转矩阵，形状为 (3, 3)
        const Eigen::Vector3d& t = x_buf_full[i].p;  // 平移向量，形状为 (3,)
        float loc_xyz[3];  // 体素坐标
        
        // 遍历当前帧的所有点
        for (PointType& pc : pl_fulls_full[i]->points) {
            // 将点从局部坐标系变换到世界坐标系
            Eigen::Vector3d pvec_orig(pc.x, pc.y, pc.z);  // 原始点坐标，形状为 (3,)
            Eigen::Vector3d pvec_tran = R * pvec_orig + t; // 变换后的点坐标，形状为 (3,)
            
            // 计算点所在的体素索引
            // 公式: voxel_idx = floor(coord / vox_size)
            for (int j = 0; j < 3; ++j) {
                loc_xyz[j] = pvec_tran[j] / vox;
                if (loc_xyz[j] < 0.0f) loc_xyz[j] -= 1.0f;  // 处理负坐标的向下取整
            }
            
            // 构建体素位置键值
            VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
            
            // 将变换后的点添加到对应体素中
            grid_map_[position].push_back(pvec_tran);
            // 记录当前帧占据的体素
            per_frame_voxels[i].insert(VOXEL_LOC{(int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]});
            ++total_points;
        }
    }


    // ========== 第二步：为每张图像建立时间窗口内的体素索引 ==========
    const double half_w = 0.5;  // 时间窗口半宽（单位：秒），即 ±0.5s 范围
    
    // 提取所有点云帧的时间戳
    std::vector<double> pcd_ts;
    pcd_ts.reserve(x_buf_full.size());
    for (const auto& kv : x_buf_full) pcd_ts.push_back(kv.t);

    // 清空并预分配体素索引容器
    all_voxel_ids_.clear();
    all_voxel_ids_.reserve(images_ids_.size());

    // 遍历每张图像，查找其时间窗口内的所有体素
    for (const double& img_id : images_ids_) {
        double t_img = 0.0;
        std::string img_name_str = std::to_string(img_id);
        
        // 从图像名称解析时间戳
        if (!parseTimestampFromName(img_name_str, t_img)) {
            std::cerr << "[buildGridMap] bad image id in images_ids_: " << img_name_str << "\n";
            all_voxel_ids_.emplace_back();  // 添加空向量作为占位
            continue;
        }

        // 计算时间窗口边界 [t0, t1]
        const double t0 = t_img - half_w;  // 时间窗口起始
        const double t1 = t_img + half_w;  // 时间窗口结束
        
        // 使用二分查找定位时间窗口内的点云帧范围
        auto itL = std::lower_bound(pcd_ts.begin(), pcd_ts.end(), t0);  // 第一个 >= t0 的位置
        auto itR = std::upper_bound(pcd_ts.begin(), pcd_ts.end(), t1);  // 第一个 > t1 的位置

        // 合并时间窗口内所有帧的体素集合
        std::set<VOXEL_LOC> voxels_set;
        for (auto it = itL; it != itR; ++it) {
            const size_t idx = static_cast<size_t>(it - pcd_ts.begin());
            if (idx >= per_frame_voxels.size()) continue;
            // 将该帧的体素集合并入总集合
            voxels_set.insert(per_frame_voxels[idx].begin(), per_frame_voxels[idx].end());
        }

        // 将set转换为vector并存储
        std::vector<VOXEL_LOC> one_vox;
        one_vox.reserve(voxels_set.size());
        for (const auto& v : voxels_set) one_vox.push_back(v);
        all_voxel_ids_.push_back(std::move(one_vox));
    }
    
    // 输出统计信息
    std::cout << "[buildGridMap] built global world cloud points=" << total_points
              << " from pcds=" << pcd_ts.size() << "\n";
    std::cout << "[buildGridMap] voxel ids: images=" << images_ids_.size()
              << ", merged window=±" << half_w << "s, vox_size=" << vox << "\n";
}

void LvbaSystem::saveTrackFeaturesOnImages()
{
    if (images_ids_.empty() || all_keypoints_.empty()) {
        std::cerr << "[TrackFeature] skip: empty images/keypoints.\n";
        return;
    }
    if (all_keypoints_.size() != images_ids_.size()) {
        std::cerr << "[TrackFeature] size mismatch: keypoints=" << all_keypoints_.size()
                  << " images=" << images_ids_.size() << "\n";
        return;
    }

    std::vector<std::vector<char>> used(all_keypoints_.size());
    for (size_t i = 0; i < all_keypoints_.size(); ++i) {
        used[i].assign(all_keypoints_[i].size(), 0);
    }

    for (const auto& tr : tracks_) {
        for (int idx_in_obs : tr.inlier_indices) {
            if (idx_in_obs < 0 || idx_in_obs >= (int)tr.observations.size()) continue;
            const auto& obs = tr.observations[idx_in_obs];
            const int cam_id = obs.first;
            const int kp_id = obs.second;
            if (cam_id < 0 || cam_id >= (int)used.size()) continue;
            if (kp_id < 0 || kp_id >= (int)used[cam_id].size()) continue;
            used[cam_id][kp_id] = 1;
        }
    }

    const std::string out_dir = dataset_path_ + "track_features/";
    if (!fs::exists(out_dir)) fs::create_directories(out_dir);

    size_t total_drawn = 0;
    size_t total_sift = 0;
    for (size_t i = 0; i < images_ids_.size(); ++i) {
        const double img_id = images_ids_[i];
        cv::Mat img = cv::imread(getImagePath(img_id), cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "[TrackFeature] cannot read image: " << img_id << "\n";
            continue;
        }

        if (img.cols != image_width_ || img.rows != image_height_) {
            cv::resize(img, img, cv::Size(img.cols * scale_, img.rows * scale_), 0, 0, cv::INTER_LINEAR);
        }

        const auto& kpts = all_keypoints_[i];
        for (const auto& kp : kpts) {
            cv::circle(img, cv::Point2f(kp.x, kp.y), 2, CV_RGB(255,0,0), -1, cv::LINE_AA);
        }

        size_t count = 0;
        for (size_t k = 0; k < used[i].size(); ++k) {
            if (!used[i][k]) continue;
            const auto& kp = all_keypoints_[i][k];
            cv::circle(img, cv::Point2f(kp.x, kp.y), 2, CV_RGB(0,255,0), -1, cv::LINE_AA);
            ++count;
        }
        total_drawn += count;
        total_sift += kpts.size();

        const std::string text = "sift=" + std::to_string(kpts.size()) + " track=" + std::to_string(count);
        cv::putText(img, text, cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    CV_RGB(255,255,255), 2, cv::LINE_AA);
        cv::putText(img, text, cv::Point(12, 24), cv::FONT_HERSHEY_SIMPLEX, 0.6,
                    CV_RGB(0,0,0), 1, cv::LINE_AA);

        const std::string out_path = out_dir + std::to_string(img_id) + ".png";
        if (!cv::imwrite(out_path, img)) {
            std::cerr << "[TrackFeature] failed to write: " << out_path << "\n";
        }

        std::cout << "[TrackFeature] img_id=" << img_id
                  << " sift=" << kpts.size()
                  << " track=" << count << "\n";
    }

    std::cout << "[TrackFeature] saved images to " << out_dir
              << " total_sift=" << total_sift
              << " total_track=" << total_drawn << "\n";
}


/**
 * [功能描述]：使用Ceres Solver进行视觉-LiDAR联合优化，同时优化相机位姿和3D地图点。
 *            该函数结合视觉重投影残差和LiDAR点-面约束，实现高精度的位姿估计。
 *            优化变量包括：相机位姿（四元数+平移）和3D地图点坐标。
 * @return 无返回值，优化结果存储在Rcw_all_optimized_、tcw_all_optimized_和tracks_中
 */
void LvbaSystem::optimizeCameraPoses()
{
    // ========== 第一步：基本数据检查 ==========
    const int M = static_cast<int>(Rcw_all_.size());  // 相机数量
    if (M == 0 || (int)tcw_all_.size() != M)
        throw std::runtime_error("optimizeCamPoses: Rcw_all_/tcw_all_ size mismatch or empty.");
    if ((int)all_keypoints_.size() != M)
        throw std::runtime_error("optimizeCamPoses: all_keypoints_.size() must equal #cameras.");
    if (tracks_.empty())
        throw std::runtime_error("optimizeCamPoses: tracks_ is empty. Run BuildTracksAndFuse3D() first.");

    // ========== 第二步：筛选可用的tracks ==========
    // 条件：观测数 >= 阈值，融合点非零且有限
    std::vector<int> track_ids; track_ids.reserve(tracks_.size());
    for (int i = 0; i < (int)tracks_.size(); ++i) {
        const auto& tr = tracks_[i];
        // [建议] 这里可以稍微提高门槛，比如 obser_thr_ 至少为 3，或者检查 rms
        if (tr.observations.size() >= obser_thr_ && !tr.Xw_fused.isZero(1e-12) && tr.Xw_fused.allFinite()) {
            track_ids.push_back(i);
        }
    }
    
    if (track_ids.empty()) {
        std::cerr << "[optimizeCamPoses] Warning: no usable tracks found!" << std::endl;
        return;
    }

    const int Npts = (int)track_ids.size();  // 可用的3D点数量
    std::cout << "[optimizeCamPoses] usable tracks = " << Npts << ", cameras = " << M << std::endl;

    // ========== 第三步：构建LiDAR体素地图用于点-面约束 ==========
    // 读取体素化参数
    const double surf_voxel_size = dataset_io_->stage2_root_voxel_size_;  // 体素大小
    const float surf_eigen_thr = dataset_io_->stage2_eigen_ratio_array_[0];  // 特征值比例阈值

    const auto& pl_fulls = dataset_io_->pl_fulls_;    // 所有帧的点云
    const auto& x_buf_full = dataset_io_->x_buf_;     // 所有帧的位姿
    const int total_size = static_cast<int>(std::min(pl_fulls.size(), x_buf_full.size()));
    if (total_size == 0) {
        std::cerr << "[optimizeCamPoses] empty pl_fulls/x_buf, skip." << std::endl;
        return;
    }

    // ---------- 3.1 构建锚点帧（窗口合并） ----------
    std::vector<IMUST> anchor_poses;                              // 锚点帧位姿
    std::vector<pcl::PointCloud<PointType>::Ptr> anchor_clouds;   // 锚点帧点云

    const int window_size = dataset_io_->window_ba_size_;     // 窗口大小
    const double anchor_leaf = dataset_io_->anchor_leaf_size_; // 降采样体素大小

    anchor_poses.reserve((total_size + window_size - 1) / window_size);
    anchor_clouds.reserve((total_size + window_size - 1) / window_size);

    // 滑动窗口合并点云
    for (int start = 0; start < total_size; start += window_size) {
        const int end = std::min(start + window_size, total_size);
        const int curr_win = end - start;
        if (curr_win <= 0) break;

        pcl::PointCloud<PointType>::Ptr merged(new pcl::PointCloud<PointType>());
        const IMUST anchor_pose = x_buf_full[start];  // 锚点取窗口第一帧

        // 将窗口内所有帧的点云变换到锚点帧坐标系并合并
        for (int j = start; j < end; ++j) {
            pcl::PointCloud<PointType> tmp = *pl_fulls[j];
            IMUST rel;
            // 计算相对位姿：T_anchor_current
            rel.R = anchor_pose.R.transpose() * x_buf_full[j].R;
            rel.p = anchor_pose.R.transpose() * (x_buf_full[j].p - anchor_pose.p);
            pl_transform(tmp, rel);
            *merged += tmp;
        }

        // 降采样并保存
        down_sampling_voxel2(*merged, anchor_leaf);
        anchor_poses.push_back(anchor_pose);
        anchor_clouds.push_back(merged);
    }

    const int anchor_size = static_cast<int>(std::min(anchor_poses.size(), anchor_clouds.size()));
    if (anchor_size == 0) {
        std::cerr << "[optimizeCamPoses] empty anchor_poses/anchor_clouds, skip." << std::endl;
        return;
    }

    // ---------- 3.2 构建体素地图 ----------
    // surf_map: 体素位置 -> 八叉树根节点
    std::unordered_map<VOXEL_LOC, OCTO_TREE_ROOT*> surf_map;
    for (int j = 0; j < anchor_size; ++j) {
        cut_voxel(surf_map, *anchor_clouds[j], anchor_poses[j], j, anchor_size,
                  surf_voxel_size, surf_eigen_thr);
    }
    printf("surf_map.size(): %zu\n", surf_map.size());
    
    // 根据当前位姿重新切割体素
    for (auto& kv : surf_map) {
        if (kv.second != nullptr) kv.second->recut(anchor_poses);
    }
    printf("After recut, surf_map.size(): %zu\n", surf_map.size());

    // ========== 第四步：初始化优化变量 ==========
    // qs[k]: 第k个相机的旋转四元数 [w, x, y, z]
    std::vector<std::array<double,4>> qs(M);
    // ts[k]: 第k个相机的平移向量 [x, y, z]
    std::vector<std::array<double,3>> ts(M);
    
    // ---------- 4.1 初始化相机位姿 ----------
    for (int k = 0; k < M; ++k) {
        Eigen::Quaterniond q_eig(Rcw_all_optimized_[k]); q_eig.normalize();
        qs[k] = { q_eig.w(), q_eig.x(), q_eig.y(), q_eig.z() };
        ts[k] = { tcw_all_optimized_[k].x(), tcw_all_optimized_[k].y(), tcw_all_optimized_[k].z() };
    }
    
    // ---------- 4.2 初始化3D点坐标 ----------
    // Xs[pi]: 第pi个3D点的世界坐标 [x, y, z]
    std::vector<std::array<double,3>> Xs(Npts);
    for (int pi = 0; pi < Npts; ++pi) {
        const auto& X = tracks_[ track_ids[pi] ].Xw_fused;
        Xs[pi] = { X.x(), X.y(), X.z() };
    }

    // ========== 第五步：计算每个3D点的平面约束 ==========
    // plane_n[pi]: 第pi个点所在平面的法向量，形状为 (3,)
    // plane_d[pi]: 平面方程 n·X + d = 0 中的d
    std::vector<Eigen::Vector3d> plane_n(Npts, Eigen::Vector3d::Zero());
    std::vector<double>          plane_d(Npts, 0.0);

    // Lambda函数：为每个3D点查找对应的LiDAR平面约束
    auto recompute_local_planes = [&](){
        for (int pi = 0; pi < Npts; ++pi)
        {
            // 获取当前3D点坐标
            Eigen::Vector3d X(Xs[pi][0], Xs[pi][1], Xs[pi][2]);
            if (!X.allFinite()) { // 检查点是否有效
                plane_n[pi].setZero(); plane_d[pi] = 0.0; continue;
            }

            // 计算点所在的体素索引
            float loc_xyz[3];
            for (int j = 0; j < 3; ++j) {
                loc_xyz[j] = X[j] / surf_voxel_size;
                if (loc_xyz[j] < 0) loc_xyz[j] -= 1.0f;
            }
            VOXEL_LOC key((int64_t)loc_xyz[0], (int64_t)loc_xyz[1], (int64_t)loc_xyz[2]);
            
            // 在体素地图中查找
            auto it = surf_map.find(key);
            if (it == surf_map.end() || it->second == nullptr) {
                plane_n[pi].setZero(); plane_d[pi] = 0.0; continue;
            }

            // 在八叉树中查找对应的叶子节点
            OCTO_TREE_NODE* node = it->second->findCorrespondPoint(X);
            if (node == nullptr || node->octo_state != PLANE) {
                plane_n[pi].setZero(); plane_d[pi] = 0.0; continue;
            }

            // 检查平面参数是否有效（法向量非NaN、非零，中心点有效）
            if (!node->direct.allFinite() || node->direct.norm() < 1e-6 || !node->center.allFinite()) {
                plane_n[pi].setZero(); plane_d[pi] = 0.0; continue;
            }

            // 提取平面参数：n·X + d = 0
            Eigen::Vector3d n = node->direct;
            n.normalize();
            plane_n[pi] = n;
            plane_d[pi] = -n.dot(node->center);  // d = -n·center
        }
    };

    // 执行平面计算
    recompute_local_planes();
    // 释放体素地图内存
    for (auto& kv : surf_map) delete kv.second;
    
    // ========== 第六步：构建Ceres优化问题 ==========
    ceres::Problem problem;
    ceres::Solver::Options options;
    options.max_num_iterations = 50;  // 最大迭代次数
    options.linear_solver_type = ceres::DENSE_SCHUR;  // 使用Schur消元求解
    options.num_threads = std::max(1u, std::thread::hardware_concurrency());  // 多线程
    options.minimizer_progress_to_stdout = true;  // 打印优化进度

    // ---------- 6.1 添加相机位姿参数块 ----------
    for (int k = 0; k < M; ++k) {
        // 四元数使用EigenQuaternionManifold流形约束（保持单位四元数）
        problem.AddParameterBlock(qs[k].data(), 4, new ceres::EigenQuaternionManifold());
        problem.AddParameterBlock(ts[k].data(), 3);
    }
    // 固定第一个相机位姿作为参考（消除规范自由度）
    problem.SetParameterBlockConstant(qs[0].data());
    problem.SetParameterBlockConstant(ts[0].data());

    // ---------- 6.2 定义鲁棒核函数 ----------
    ceres::LossFunction* loss_function_reproj = new ceres::HuberLoss(1.0);  // 重投影残差的Huber核
    ceres::LossFunction* loss_function_plane  = new ceres::HuberLoss(0.1);  // 点-面残差的Huber核

    // 标记每个3D点是否参与优化（有有效平面约束）
    std::vector<bool> point_is_valid(Npts, false);

    // ---------- 6.3 添加残差块 ----------
    const double sigma_px = 0.5; // 特征点测量噪声标准差（像素）

    for (int pi = 0; pi < Npts; ++pi) {
        
        const Eigen::Vector3d& n = plane_n[pi];  // 平面法向量
        const double d = plane_d[pi];             // 平面参数d

        // 严格筛选：如果没有有效的平面约束，跳过该点
        bool has_valid_plane = (n.allFinite() && std::isfinite(d) && !n.isZero(1e-6));
        
        if (!has_valid_plane) {
            // 没找到平面约束 -> 该点不参与优化
            point_is_valid[pi] = false; 
            continue; 
        }

        // 通过筛选，标记为有效
        point_is_valid[pi] = true;

        // (1) 添加3D点参数块
        problem.AddParameterBlock(Xs[pi].data(), 3);

        // (2) 添加视觉重投影残差
        const int tid = track_ids[pi];
        const auto& tr = tracks_[tid]; 
        std::unordered_set<int> seen;  // 用于去重
        
        // 遍历该track的所有inlier观测
        for (int idx_in_obs : tr.inlier_indices) {
            if (idx_in_obs < 0 || idx_in_obs >= (int)tr.observations.size()) continue;
            if (!seen.insert(idx_in_obs).second) continue;  // 跳过重复
    
            const auto& obs = tr.observations[idx_in_obs];
            const int cam_id = obs.first;   // 相机索引
            const int kp_id  = obs.second;  // 特征点索引
            if (cam_id < 0 || cam_id >= M) continue;
    
            // 获取观测的像素坐标
            const double u = all_keypoints_[cam_id][kp_id].x;
            const double v = all_keypoints_[cam_id][kp_id].y;

            // 创建带畸变的重投影误差代价函数
            // 参数：观测坐标(u,v)、相机内参、畸变系数、噪声标准差
            ceres::CostFunction* cost = ReprojErrorWhitenedDistorted::Create(
                    u, v, fx_, fy_, cx_, cy_, d0_, d1_, d2_, d3_, sigma_px, sigma_px);
            
            // 添加残差块，优化变量：相机旋转、相机平移、3D点坐标
            problem.AddResidualBlock(cost, nullptr,
                                     qs[cam_id].data(), ts[cam_id].data(), Xs[pi].data());
        }

        // (3) 添加点-面残差（LiDAR约束）
        // 约束3D点位于LiDAR检测到的平面上
        double sigma_plane = 0.01;  // 平面约束的噪声标准差（米）
        ceres::CostFunction* plane_cost = PointPlaneErrorWhitened::Create(n, d, sigma_plane);
        problem.AddResidualBlock(plane_cost, nullptr, Xs[pi].data());
    }

    // ========== 第七步：执行优化求解 ==========
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    std::cout << "[optimizeCamPoses] " << summary.BriefReport() << std::endl;

    // 检查求解是否成功
    if (summary.termination_type == ceres::FAILURE) {
        std::cerr << "[optimizeCamPoses] Solver failed!" << std::endl;
        return;
    }

    // ========== 第八步：回写优化结果 ==========
    // ---------- 8.1 回写相机位姿 ----------
    for (int k = 0; k < M; ++k) {
        // 从四元数恢复旋转矩阵
        Eigen::Quaterniond q_eig(qs[k][0], qs[k][1], qs[k][2], qs[k][3]);
        q_eig.normalize();
        Rcw_all_optimized_[k] = q_eig.toRotationMatrix();
        // 回写平移向量
        tcw_all_optimized_[k] = Eigen::Vector3d(ts[k][0], ts[k][1], ts[k][2]);
    }

    // ---------- 8.2 回写3D点坐标 ----------
    // 只回写参与优化的有效点
    int valid_cnt = 0;
    for (int pi = 0; pi < Npts; ++pi) {
        if (point_is_valid[pi]) {
            // 更新优化后的3D点坐标
            Eigen::Vector3d X_new(Xs[pi][0], Xs[pi][1], Xs[pi][2]);
            tracks_[ track_ids[pi] ].Xw_fused = X_new;
            valid_cnt++;
        } 
    }
    
    std::cout << "[optimizeCamPoses] Points kept: " << valid_cnt << " / " << Npts << std::endl;

    std::cout << "[optimizeCamPoses] done." << std::endl;
}

/**
 * [功能描述]：可视化重投影误差，对比优化前后的效果。
 *            将3D地图点分别用优化前和优化后的位姿投影到图像上，
 *            与实际观测的特征点位置进行比较，并计算重投影误差统计量。
 * @return 无返回值，可视化结果保存到 dataset_path_/reproj/ 目录
 */
void LvbaSystem::visualizeProj() {

    namespace fs = std::filesystem;

    // ========== 第一步：定义辅助Lambda函数 ==========
    
    // ---------- drawCross: 在图像上绘制十字标记 ----------
    // @param img: 目标图像
    // @param p: 十字中心点坐标
    // @param size: 十字臂长度（像素）
    // @param thickness: 线宽
    // @param color: 颜色
    auto drawCross = [](cv::Mat& img, const cv::Point2d& p, int size, int thickness, const cv::Scalar& color) {
        cv::line(img, cv::Point2d(p.x - size, p.y), cv::Point2d(p.x + size, p.y), color, thickness, cv::LINE_AA);
        cv::line(img, cv::Point2d(p.x, p.y - size), cv::Point2d(p.x, p.y + size), color, thickness, cv::LINE_AA);
    };
    
    // ---------- putTextShadow: 在图像上绘制文字（带可选阴影） ----------
    auto putTextShadow = [](cv::Mat& img, const std::string& text, cv::Point org, double scale=0.45, int thick=1, cv::Scalar color=CV_RGB(0,0,0)) {
        // cv::putText(img, text, org + cv::Point(1,1), cv::FONT_HERSHEY_SIMPLEX, scale, CV_RGB(0,0,0), thick+2, cv::LINE_AA);
        cv::putText(img, text, org, cv::FONT_HERSHEY_SIMPLEX, scale, color, thick, cv::LINE_AA);
    };
    
    // ---------- projectWithDistortion: 带畸变的3D点投影 ----------
    // 将世界坐标系的3D点投影到图像平面，考虑相机畸变
    // @param Pw: 世界坐标系下的3D点，形状为 (3,)
    // @param Rcw: 世界到相机的旋转矩阵，形状为 (3, 3)
    // @param tcw: 世界到相机的平移向量，形状为 (3,)
    // @param uv_out: 输出的像素坐标
    // @return: 投影成功返回true，失败返回false
    auto projectWithDistortion = [&](const Eigen::Vector3d& Pw,
                                     const Eigen::Matrix3d& Rcw,
                                     const Eigen::Vector3d& tcw,
                                     cv::Point2d& uv_out) -> bool {
        // 将点从世界坐标系变换到相机坐标系
        const Eigen::Vector3d Pc = Rcw * Pw + tcw;
        const double X = Pc.x(), Y = Pc.y(), Z = Pc.z();
        // 检查深度是否有效
        if (!(Z > 1e-12) || !std::isfinite(X) || !std::isfinite(Y) || !std::isfinite(Z)) return false;
        
        // 计算归一化平面坐标                        
        const double x = X / Z, y = Y / Z;
        
        // 应用Brown-Conrady畸变模型
        const double r2 = x*x + y*y, r4 = r2*r2;
        // 径向畸变: radial = 1 + k1*r2 + k2*r4
        const double radial = 1.0 + d0_ * r2 + d1_ * r4;
        // 切向畸变
        const double x_t = 2.0*d2_*x*y + d3_*(r2 + 2.0*x*x);
        const double y_t = d2_*(r2 + 2.0*y*y) + 2.0*d3_*x*y;
        // 畸变后的归一化坐标
        const double xd = x*radial + x_t;
        const double yd = y*radial + y_t;

        // 投影到像素坐标
        const double u = fx_ * xd + cx_;
        const double v = fy_ * yd + cy_;
        if (!std::isfinite(u) || !std::isfinite(v)) return false;
        uv_out = cv::Point2d(u, v);
        return true;
    };

    // ========== 第二步：基本数据检查 ==========
    const int M = static_cast<int>(images_ids_.size());  // 图像数量
    if (M == 0) {
        std::cerr << "[visualizeProj] images_ids_ is empty.\n";
        return;
    }
    // 检查位姿数组大小是否匹配
    if (Rcw_all_.size() != (size_t)M || tcw_all_.size() != (size_t)M ||
        Rcw_all_optimized_.size() != (size_t)M || tcw_all_optimized_.size() != (size_t)M) {
        std::cerr << "[visualizeProj] pose arrays size mismatch with images_ids_.\n";
        return;
    }

    // ========== 第三步：创建输出目录 ==========
    std::string out_dir = dataset_path_ + "reproj";
    if (!fs::exists(out_dir)) fs::create_directories(out_dir);

    // 检查是否有优化后的三维点
    bool has_after_pts = false;
    if (this->tracks_.size() == this->tracks_before_.size() && !this->tracks_.empty()) {
        has_after_pts = true;
    }

    // ========== 第四步：定义可视化条目结构并按图像聚合 ==========
    // Item结构：存储每个观测点的可视化信息
    struct Item {
        cv::Point2d uv_meas;                    // 实际观测的像素坐标
        bool has_pre=false, has_post=false;    // 优化前/后投影是否有效
        cv::Point2d uv_pre, uv_post;           // 优化前/后的投影像素坐标
        double err_pre=-1, err_post=-1;        // 优化前/后的重投影误差（像素）
        int track_id=-1;                        // 所属track的ID
    };
    // per_image_items[k]: 第k张图像上的所有可视化条目
    std::vector<std::vector<Item>> per_image_items(M);

    // ========== 第五步：收集所有内点观测的重投影信息 ==========
    for (int tid = 0; tid < (int)tracks_before_.size(); ++tid) {
        const auto& tr_b = tracks_before_[tid];  // 优化前的track
        const auto& tr_a = tracks_[tid];         // 优化后的track

        // 获取优化前后的3D点坐标
        const Eigen::Vector3d Pw_pre  = tr_b.Xw_fused;   // 优化前的3D点
        const Eigen::Vector3d Pw_post = tr_a.Xw_fused;   // 优化后的3D点

        // 用于去重的集合
        std::unordered_set<int> seen;
        seen.reserve(tr_b.inlier_indices.size());

        // 遍历该track的所有内点观测
        for (int idx_in_obs : tr_b.inlier_indices) {
            if (idx_in_obs < 0 || idx_in_obs >= (int)tr_b.observations.size()) continue;
            if (!seen.insert(idx_in_obs).second) continue;  // 跳过重复

            const auto& obs = tr_b.observations[idx_in_obs];
            const int cam_id = obs.first;   // 相机索引
            const int kp_id  = obs.second;  // 特征点索引
            if (cam_id < 0 || cam_id >= M) continue;
            if (kp_id  < 0 || kp_id >= (int)all_keypoints_[cam_id].size()) continue;

            // 获取实际观测的像素坐标
            const double u_meas = all_keypoints_[cam_id][kp_id].x;
            const double v_meas = all_keypoints_[cam_id][kp_id].y;
            cv::Point2d uv_meas(u_meas, v_meas);

            // 计算优化前的投影（使用优化前的位姿和3D点）
            cv::Point2d uv_pre;
            bool ok_pre = projectWithDistortion(Pw_pre, Rcw_all_[cam_id], tcw_all_[cam_id], uv_pre);
            
            // 计算优化后的投影（使用优化后的位姿和3D点）
            cv::Point2d uv_post;
            bool ok_post = projectWithDistortion(Pw_post, Rcw_all_optimized_[cam_id], tcw_all_optimized_[cam_id], uv_post);

            // 构建可视化条目
            Item it;
            it.uv_meas = uv_meas;
            it.has_pre = ok_pre;
            it.has_post = ok_post;
            it.uv_pre = uv_pre;
            it.uv_post = uv_post;
            // 计算重投影误差（欧氏距离）
            it.err_pre  = ok_pre  ? cv::norm(uv_pre  - uv_meas) : -1.0;
            it.err_post = ok_post ? cv::norm(uv_post - uv_meas) : -1.0;
            it.track_id = tid;

            per_image_items[cam_id].push_back(std::move(it));
        }
    }

    // ========== 第六步：绘制可视化图像并保存 ==========
    double global_err_pre = 0.0, global_err_post = 0.0;  // 全局误差累计
    int global_cnt = 0;
    
    for (int k = 0; k < M; ++k) {
        // 读取原始图像
        const double img_id = images_ids_[k];
        const std::string img_path = getImagePath(img_id);
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "[visualizeProj] cannot read image: " << img_path << "\n";
            continue;
        }

        // 当前图像的误差统计
        double sum_pre = 0.0, sum_post = 0.0;
        int cnt_pre = 0, cnt_post = 0;

        // 遍历当前图像上的所有可视化条目
        for (const auto& it : per_image_items[k]) {
            // 绘制实际观测点：绿色十字
            drawCross(img, it.uv_meas, 5, 1, CV_RGB(0,255,0));

            // 绘制优化前的投影点：蓝色圆点
            if (it.has_pre) {
                cv::circle(img, it.uv_pre, 2, CV_RGB(0,128,255), -1, cv::LINE_AA);
                // cv::line(img, it.uv_pre, it.uv_meas, CV_RGB(0,128,255), 1, cv::LINE_AA);
                sum_pre += it.err_pre; cnt_pre++;
            }
            // 绘制优化后的投影点：红色方块
            if (it.has_post) {
                cv::rectangle(img, cv::Point(it.uv_post.x-1, it.uv_post.y-1), cv::Point(it.uv_post.x+1, it.uv_post.y+1), CV_RGB(255,0,0), -1, cv::LINE_AA);
                // cv::line(img, it.uv_post, it.uv_meas, CV_RGB(255,0,0), 1, cv::LINE_AA);
                sum_post += it.err_post; cnt_post++;
            }
        }

        // 计算当前图像的平均重投影误差
        const double mean_pre  = (cnt_pre  > 0) ? (sum_pre  / cnt_pre)  : -1.0;
        const double mean_post = (cnt_post > 0) ? (sum_post / cnt_post) : -1.0;
        global_cnt++; 
        global_err_pre += mean_pre; 
        global_err_post += mean_post;
        
        // 在图像左上角绘制统计信息和图例
        {
            std::ostringstream head;
            head.setf(std::ios::fixed); head.precision(3);
            head << "img_id=" << img_id
                 << "  N=" << per_image_items[k].size()
                 << "  mean_pre=" << mean_pre
                 << "  mean_post=" << mean_post;
            putTextShadow(img, head.str(), cv::Point(12, 24), 0.9 * scale_, 1, CV_RGB(0,0,0));
            // 图例说明
            putTextShadow(img, "meas: green cross",      cv::Point(12, 48), 0.55 * scale_, 1, CV_RGB(0,255,0));   // 观测：绿色十字
            putTextShadow(img, "pre:  blue dot",  cv::Point(12, 68), 0.55 * scale_, 1, CV_RGB(0,0,255));          // 优化前：蓝点
            putTextShadow(img, "post: red rectangle", cv::Point(12, 88), 0.55 * scale_, 1, CV_RGB(255,0,0));      // 优化后：红方块
        }

        // 保存可视化图像
        char out_name[512];
        std::snprintf(out_name, sizeof(out_name), "%s/vis_%08.0f.png", out_dir.c_str(), img_id);
        if (!cv::imwrite(out_name, img)) {
            std::cerr << "[visualizeProj] failed to write: " << out_name << "\n";
        }
    }
    
    // ========== 第七步：输出全局统计信息 ==========
    global_err_pre /= global_cnt;   // 全局平均优化前误差
    global_err_post /= global_cnt;  // 全局平均优化后误差
    std::cout << "[visualizeProj] global mean pre: " << global_err_pre << "\n";
    std::cout << "[visualizeProj] global mean post: " << global_err_post << "\n";

    std::cout << "[visualizeProj] done. saved to: " << out_dir << std::endl;
    
    // 调用对比可视化函数
    std::string dir = dataset_path_ + "colmap/colored_merged.pcd";
    VisualizeOptComparison(images_ids_, true, dir);
}

void LvbaSystem::showTracksComparePCL() 
{
    std::cout << "[Visualizer] Preparing data..." << std::endl;

    pcl::PointCloud<pcl::PointXYZ>::Ptr track_viz_before(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr track_viz_after (new pcl::PointCloud<pcl::PointXYZ>());
    
    track_viz_before->reserve(tracks_before_.size());
    for (const auto& tr : tracks_before_) {
        const auto& X = tr.Xw_fused;
        if (X.allFinite()) track_viz_before->emplace_back((float)X.x(), (float)X.y(), (float)X.z());
    }
    
    track_viz_after->reserve(tracks_.size());
    for (const auto& tr : tracks_) {
        const auto& X = tr.Xw_fused;
        if (X.allFinite()) track_viz_after->emplace_back((float)X.x(), (float)X.y(), (float)X.z());
    }

    std::cout << "[Visualizer] Before: " << track_viz_before->size() << " | After: " << track_viz_after->size() << std::endl;

    std::string target_frame_id = "map"; 
    ros::Time current_time = ros::Time::now();
    
    if (track_viz_before->size() > 0) {
        sensor_msgs::PointCloud2 msg_before;
        pcl::toROSMsg(*track_viz_before, msg_before);
        msg_before.header.frame_id = target_frame_id;
        msg_before.header.stamp = current_time;
        pub_cloud_before_.publish(msg_before);
    }

    if (track_viz_after->size() > 0) {
        sensor_msgs::PointCloud2 msg_after;
        pcl::toROSMsg(*track_viz_after, msg_after);
        msg_after.header.frame_id = target_frame_id;
        msg_after.header.stamp = current_time;
        pub_cloud_after_.publish(msg_after);
    }
}

void LvbaSystem::drawAndSaveMatchesGPU(
    const std::string& out_dir,
    int id1, int id2,
    const cv::Mat& img1, const cv::Mat& img2,
    const std::vector<SiftGPU::SiftKeypoint>& kpts1,
    const std::vector<SiftGPU::SiftKeypoint>& kpts2,
    const std::vector<std::pair<int,int>>& matches) {

    namespace fs = std::filesystem;
    fs::create_directories(out_dir);

    // 拼接画布
    int H = std::max(img1.rows, img2.rows);
    int W = img1.cols + img2.cols;
    cv::Mat canvas(H, W, CV_8UC3, cv::Scalar(20,20,20));
    img1.copyTo(canvas(cv::Rect(0,0,img1.cols,img1.rows)));
    img2.copyTo(canvas(cv::Rect(img1.cols,0,img2.cols,img2.rows)));

    // 随机颜色
    cv::RNG rng(12345);
    auto randColor = [&](){ return cv::Scalar(rng.uniform(64,255),
                                                rng.uniform(64,255),
                                                rng.uniform(64,255)); };

    for (auto& m : matches) {
        int i1 = m.first, i2 = m.second;
        if (i1<0 || i1>=(int)kpts1.size() || i2<0 || i2>=(int)kpts2.size()) continue;

        cv::Point2f p1(kpts1[i1].x, kpts1[i1].y);
        cv::Point2f p2(kpts2[i2].x + img1.cols, kpts2[i2].y);

        auto col = randColor();
        cv::circle(canvas, p1, 3, col, -1, cv::LINE_AA);
        cv::circle(canvas, p2, 3, col, -1, cv::LINE_AA);
        cv::line(canvas, p1, p2, col, 1, cv::LINE_AA);
    }
    std::cout << " Drawed : " << id1 << " - " << id2
              << " | matches: " << matches.size() << std::endl;
    std::string save_path = out_dir + "/" + std::to_string(id1+1) + "_" + std::to_string(id2+1) + "_matches_nums:" + std::to_string(matches.size())+".jpg";
    cv::imwrite(save_path, canvas);
}

bool LvbaSystem::ProjectToImage(
    const Eigen::Matrix3d& Rcw, const Eigen::Vector3d& tcw,
    const Eigen::Vector3d& Xw,
    double* u, double* v, double* Zc) const
{
    // 世界->相机
    const Eigen::Vector3d Xc = Rcw * Xw + tcw;
    const double z = Xc.z();
    if (Zc) *Zc = z;
    if (z <= 1e-6) return false;  // 在相机后方/接近零深度，丢弃

    // 归一化坐标
    const double xn = Xc.x() / z;
    const double yn = Xc.y() / z;

    // Brown-Conrady: k1,k2,p1,p2
    const double k1 = d0_, k2 = d1_, p1 = d2_, p2 = d3_;
    const double r2  = xn*xn + yn*yn;
    const double r4  = r2 * r2;
    const double radial = 1.0 + k1 * r2 + k2 * r4;

    // 切向
    const double x_tan = 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn);
    const double y_tan = p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn;

    // 畸变后归一化坐标
    const double xdist = xn * radial + x_tan;
    const double ydist = yn * radial + y_tan;

    // 像素坐标
    const double uu = fx_ * xdist + cx_;
    const double vv = fy_ * ydist + cy_;

    if (u) *u = uu;
    if (v) *v = vv;
    return std::isfinite(uu) && std::isfinite(vv);
}


void LvbaSystem::VisualizeOptComparison(
    const std::vector<double>& image_ids,
    bool save_merged_pcd,
    const std::string& merged_pcd_path)
{
    const auto& pl_fulls = dataset_io_->pl_fulls_;
    const auto& x_buf_opt = dataset_io_->x_buf_;
    const auto& x_buf_bef = dataset_io_->x_buf_before_;

    // 聚合的彩色点云
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged(new pcl::PointCloud<pcl::PointXYZRGB>());
    merged->reserve(3000000); // 预留一些容量，可按需调整
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_b(new pcl::PointCloud<pcl::PointXYZRGB>());
    merged_b->reserve(3000000); // 预留一些容量，可按需调整

    // 遍历每一帧
    // fout_poses_before.open(dataset_path_ + "colmap/before_sparse/images.txt", std::ios::out);
    // fout_poses_after.open(dataset_path_ + "colmap/after_sparse/images.txt", std::ios::out);

    for (size_t k = 0; k < image_ids.size(); ++k) {
        const double img_id = image_ids[k];
        const std::string img_path = getImagePath(img_id);        

        // 读图（BGR）
        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img.empty()) {
            std::cerr << "[Colorize] Failed to load image: " << img_path << "\n";
            continue;
        }
        if (img.cols != image_width_ || img.rows != image_height_) 
        {
            cv::resize(img, img, cv::Size(img.cols * scale_, img.rows * scale_), 0, 0, cv::INTER_LINEAR);        
        }
        const int W = img.cols, H = img.rows;

        //-----------------------多帧合并（内存点云）----------------------//
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_w_all_opt(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_w_all_orig(new pcl::PointCloud<pcl::PointXYZ>());

        for (size_t idx = 0; idx < x_buf_opt.size(); ++idx) {
            if (std::fabs(x_buf_opt[idx].t - img_id) > 0.5) {
                continue;
            }
            if (idx >= pl_fulls.size()) continue;
            const auto& pl_body = pl_fulls[idx];
            const IMUST& pose_opt = x_buf_opt[idx];
            const IMUST& pose_bef = x_buf_bef[idx];

            for (const auto& pb : pl_body->points) {
                Eigen::Vector3d Xw_opt = pose_opt.R * Eigen::Vector3d(pb.x, pb.y, pb.z) + pose_opt.p;
                cloud_w_all_opt->emplace_back(static_cast<float>(Xw_opt.x()),
                                              static_cast<float>(Xw_opt.y()),
                                              static_cast<float>(Xw_opt.z()));
                Eigen::Vector3d Xw_orig = pose_bef.R * Eigen::Vector3d(pb.x, pb.y, pb.z) + pose_bef.p;
                cloud_w_all_orig->emplace_back(static_cast<float>(Xw_orig.x()),
                                               static_cast<float>(Xw_orig.y()),
                                               static_cast<float>(Xw_orig.z()));
            }
        }
        if (cloud_w_all_opt->empty() || cloud_w_all_orig->empty()) {
            std::cerr << "[Colorize] skip image " << img_id << " no lidar in window\n";
            continue;
        }

        const Eigen::Matrix3d& Rcw = Rcw_all_optimized_[k];
        const Eigen::Vector3d& tcw = tcw_all_optimized_[k];
        const Eigen::Matrix3d& Rcw_b = Rcw_all_[k];
        const Eigen::Vector3d& tcw_b = tcw_all_[k];
        // colmap 格式
        Eigen::Quaterniond q(Rcw);
        Eigen::Vector3d t = tcw;
        Eigen::Quaterniond q_b(Rcw_b);
        Eigen::Vector3d t_b = tcw_b;
        // fout_poses_after << k << " "
        //           << std::fixed << std::setprecision(6)  // 保证浮点数精度为6位
        //           << q.w() << " " << q.x() << " " << q.y() << " " << q.z() << " "
        //           << t.x() << " " << t.y() << " " << t.z() << " "
        //           << 1 << " "  // CAMERA_ID (假设相机ID为1)
        //           << k << ".jpg" << std::endl;
        // fout_poses_after << "0.0 0.0 -1" << std::endl;

        // fout_poses_before << k << " "
        //           << std::fixed << std::setprecision(6)  // 保证浮点数精度为6位
        //           << q_b.w() << " " << q_b.x() << " " << q_b.y() << " " << q_b.z() << " "
        //           << t_b.x() << " " << t_b.y() << " " << t_b.z() << " "
        //           << 1 << " "  // CAMERA_ID (假设相机ID为1)
        //           << k << ".jpg" << std::endl;
        // fout_poses_before << "0.0 0.0 -1" << std::endl;

        const std::string out = dataset_path_ + "colmap/images/" + std::to_string(k) + ".jpg";

        cv::Mat undist;
        dataset_io_->undistortImage(img, undist);
        cv::imwrite(out, undist);
        // colmap 格式
        std::vector<float> zbuf(W * H, std::numeric_limits<float>::infinity());
        std::vector<pcl::PointXYZRGB> pixbuf(W * H);
        const float eps = 1e-6f;
        
        for (const auto& p : cloud_w_all_opt->points) {
            Eigen::Vector3d Xw(p.x, p.y, p.z);
        
            double u = 0, v = 0, zc = 0;
            if (!ProjectToImage(Rcw, tcw, Xw, &u, &v, &zc)) continue;
        
            int uu = static_cast<int>(std::round(u));
            int vv = static_cast<int>(std::round(v));
            if (uu < 0 || uu >= W || vv < 0 || vv >= H) continue;
        
            const int idx = vv * W + uu;
        
            // 只保留深度最近（zc越小越近）
            if (zc + eps < zbuf[idx]) {
                const cv::Vec3b bgr = img.at<cv::Vec3b>(vv, uu);
        
                pcl::PointXYZRGB cp;
                cp.x = p.x; cp.y = p.y; cp.z = p.z;
                cp.b = bgr[0]; cp.g = bgr[1]; cp.r = bgr[2];
        
                zbuf[idx]   = static_cast<float>(zc);
                pixbuf[idx] = cp;
            }
        }
        
        // 把每个像素最终留下的“最近点”收集到 colored 里
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored(new pcl::PointCloud<pcl::PointXYZRGB>());
        colored->reserve(W * H); // 粗略预留，可不写
        for (int i = 0; i < W * H; ++i) {
            if (std::isfinite(zbuf[i])) colored->push_back(pixbuf[i]);
        }

        *merged += *colored;

        std::vector<float> zbuf_b(W * H, std::numeric_limits<float>::infinity());
        std::vector<pcl::PointXYZRGB> pixbuf_b(W * H);
        const float eps_b = 1e-6f;
        for (const auto& p : cloud_w_all_orig->points) {
            Eigen::Vector3d Xw(p.x, p.y, p.z);

            double u = 0, v = 0, zc = 0;

            if (!ProjectToImage(Rcw_b, tcw_b, Xw, &u, &v, &zc)) continue;

            int uu = static_cast<int>(std::round(u));
            int vv = static_cast<int>(std::round(v));

            if (uu < 0 || uu >= W || vv < 0 || vv >= H) continue;

            const int idx = vv * W + uu;
        
            // 只保留深度最近（zc越小越近）
            if (zc + eps_b < zbuf_b[idx]) {
                const cv::Vec3b bgr = img.at<cv::Vec3b>(vv, uu);
        
                pcl::PointXYZRGB cp;
                cp.x = p.x; cp.y = p.y; cp.z = p.z;
                cp.b = bgr[0]; cp.g = bgr[1]; cp.r = bgr[2];
        
                zbuf_b[idx]   = static_cast<float>(zc);
                pixbuf_b[idx] = cp;
            }
        }
        // 把每个像素最终留下的“最近点”收集到 colored 里
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_b(new pcl::PointCloud<pcl::PointXYZRGB>());
        colored_b->reserve(W * H); // 粗略预留，可不写
        for (int i = 0; i < W * H; ++i) {
            if (std::isfinite(zbuf_b[i])) colored_b->push_back(pixbuf_b[i]);
        }
        *merged_b += *colored_b;
    }

    pub_cloud_b_ = merged_b;



    // std::cout << "[Colorize] Merged colored cloud size = " << merged->size() << "\n";
    // pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZRGB>());
    // if (merged->size() > 8000000) {
    //     pcl::VoxelGrid<pcl::PointXYZRGB> voxel_grid;
    //     voxel_grid.setInputCloud(merged);
    //     voxel_grid.setLeafSize(0.15f, 0.15f, 0.15f); 
    //     voxel_grid.filter(*cloud_downsampled);
    //     merged = cloud_downsampled;
    // }
    // std::cout << "[Colorize] Merged colored cloud size = " << merged->size() << "\n";

    pub_cloud_ = merged;

    // fout_points_after.open(dataset_path_+ "colmap/after_sparse/points3D.txt", std::ios::out);
    // for (size_t i = 0; i < merged->size(); ++i) 
    // {
    //     // std::cout << i << std::endl;
    //     const auto& point = merged->points[i];
    //     fout_points_after << i << " "
    //                 << std::fixed << std::setprecision(6)
    //                 << point.x << " " << point.y << " " << point.z << " "
    //                 << static_cast<int>(point.r) << " "
    //                 << static_cast<int>(point.g) << " "
    //                 << static_cast<int>(point.b) << " "
    //                 << 0 << std::endl;
    // }

    // std::cout << "color after done" << std::endl;
    // std::cout << "color before done" << std::endl;
    std::vector<pcl::PointCloud<PointType>::Ptr>().swap(dataset_io_->pl_fulls_);
}

std::string LvbaSystem::getImagePath(double image_id) {
  return dataset_path_ + "all_image/" + std::to_string(image_id) + ".png";
}

std::string LvbaSystem::getPcdPath(double pcd_id) {
  return dataset_path_ + "all_pcd_body/" + std::to_string(pcd_id) + ".pcd";
}

void LvbaSystem::pubRGBCloud() {

    showTracksComparePCL();

    sensor_msgs::PointCloud2 output;
    down_sampling_voxel(*pub_cloud_, 0.01);
    pcl::toROSMsg(*pub_cloud_, output);
    output.header.frame_id = "map";
    output.header.stamp = ros::Time::now();

    cloud_pub_after_.publish(output);

    sensor_msgs::PointCloud2 output_b;
    down_sampling_voxel(*pub_cloud_b_, 0.01);
    pcl::toROSMsg(*pub_cloud_b_, output_b);
    output_b.header.frame_id = "map"; 
    output_b.header.stamp = ros::Time::now();

    cloud_pub_before_.publish(output_b);
}


}
