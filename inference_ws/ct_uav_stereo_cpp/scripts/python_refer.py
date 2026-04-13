import cv2
import numpy as np
import yaml

def tune_sgbm_parameters(video_path, config_path):
    """
    Interactive tuning of SGBM parameters using trackbars
    """
    # Load stereo configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract camera parameters (cam0 = left, cam1 = right)
    cam0 = config['cam0']
    cam1 = config['cam1']
    
    # Parse intrinsics [fu, fv, cu, cv]
    intrinsics_0 = cam0['intrinsics']
    intrinsics_1 = cam1['intrinsics']
    
    K_cam0 = np.array([
        [intrinsics_0[0], 0, intrinsics_0[2]],
        [0, intrinsics_0[1], intrinsics_0[3]],
        [0, 0, 1]
    ])
    D_cam0 = np.array(cam0['distortion_coeffs'])
    
    K_cam1 = np.array([
        [intrinsics_1[0], 0, intrinsics_1[2]],
        [0, intrinsics_1[1], intrinsics_1[3]],
        [0, 0, 1]
    ])
    D_cam1 = np.array(cam1['distortion_coeffs'])
    
    # Extract extrinsics T_cam1_cam0
    T_cam1_cam0 = np.array(cam1['T_cn_cnm1'])
    R_cam1_cam0 = T_cam1_cam0[0:3, 0:3]
    t_cam1_cam0 = T_cam1_cam0[0:3, 3].reshape(3, 1)
    
    # Image size
    image_size = tuple(cam0['resolution'])
    
    # Stereo rectification using Fusiello's method
    # Reference: A.Fusiello, E. Trucco, A. Verri: A compact algorithm for rectification of stereo pairs, 1999
    
    # Projection matrices (cam0 as world frame)
    Poa = np.matrix(K_cam0) @ np.hstack((np.matrix(np.eye(3)), np.matrix(np.zeros((3,1)))))
    Pob = np.matrix(K_cam1) @ np.hstack((np.matrix(R_cam1_cam0), np.matrix(t_cam1_cam0)))
    
    # Optical centers (in cam0's coord sys)
    c1 = -np.linalg.inv(Poa[:, 0:3]) @ Poa[:, 3]
    c2 = -np.linalg.inv(Pob[:, 0:3]) @ Pob[:, 3]
    
    # Get "mean" rotation between cams
    old_z_mean = (R_cam1_cam0[2, :].flatten() + np.eye(3)[2, 0:3]) / 2.0
    v1 = c1 - c2  # new x-axis = direction of baseline
    v2 = np.cross(np.matrix(old_z_mean).flatten(), v1.flatten()).T  # new y axis orthogonal to new x and mean old z
    v3 = np.cross(v1.flatten(), v2.flatten()).T  # orthogonal to baseline and new y
    
    # Normalize
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    v3 = v3 / np.linalg.norm(v3)
    
    # Create rotation matrix
    R = np.hstack((np.hstack((v1, v2)), v3)).T
    
    # New intrinsic parameters
    A = (K_cam0 + K_cam1) / 2.0
    
    # Rectifying transforms
    Ra = R  # cam0=world, then to rectified coords
    Rb = R @ np.linalg.inv(R_cam1_cam0)  # to world then to rectified coords
    
    # Compute rectification maps
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        K_cam0, D_cam0, Ra, A, image_size, cv2.CV_16SC2
    )
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        K_cam1, D_cam1, Rb, A, image_size, cv2.CV_16SC2
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    
    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error reading video")
        return
    
    # Split stereo frame (assuming left and right are side by side)
    height, width = frame.shape[:2]
    left_frame = frame[:, :width//2]
    right_frame = frame[:, width//2:]
    
    # Rectify images
    left_rectified = cv2.remap(left_frame, map1_left, map2_left, cv2.INTER_LINEAR)
    right_rectified = cv2.remap(right_frame, map1_right, map2_right, cv2.INTER_LINEAR)
    
    left_resize = cv2.resize(left_rectified, (640, 360))
    right_resize = cv2.resize(right_rectified, (640, 360))
    
    # Create window and trackbars
    window_name = 'SGBM Tuning'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # State variables
    show_epipolar = False
    use_wls_filter = False
    
    # Trackbar callback (does nothing, just needed)
    def nothing(x):
        pass
    
    # Create trackbars for SGBM parameters
    cv2.createTrackbar('minDisparity', window_name, 0, 50, nothing)
    cv2.createTrackbar('numDisparities', window_name, 6, 16, nothing)  # Will multiply by 16
    cv2.createTrackbar('blockSize', window_name, 11, 25, nothing)
    cv2.createTrackbar('P1_multiplier', window_name, 8, 100, nothing)
    cv2.createTrackbar('P2_multiplier', window_name, 32, 200, nothing)
    cv2.createTrackbar('disp12MaxDiff', window_name, 1, 50, nothing)
    cv2.createTrackbar('uniquenessRatio', window_name, 10, 100, nothing)
    cv2.createTrackbar('speckleWindowSize', window_name, 100, 300, nothing)
    cv2.createTrackbar('speckleRange', window_name, 32, 100, nothing)
    cv2.createTrackbar('preFilterCap', window_name, 63, 127, nothing)
    
    # WLS Filter parameters
    cv2.createTrackbar('WLS_lambda', window_name, 8000, 20000, nothing)
    cv2.createTrackbar('WLS_sigma', window_name, 15, 50, nothing)
    
    print("Controls:")
    print("  - Adjust trackbars to tune SGBM parameters")
    print("  - Press 'n' for next frame")
    print("  - Press 'e' to toggle epipolar lines")
    print("  - Press 'w' to toggle WLS filter")
    print("  - Press 's' to save current parameters")
    print("  - Press 'q' to quit")
    
    def draw_epipolar_lines(img_left, img_right, num_lines=20):
        """Draw horizontal epipolar lines on rectified stereo pair"""
        img_left_copy = img_left.copy()
        img_right_copy = img_right.copy()
        
        height = img_left.shape[0]
        step = height // num_lines
        
        for i in range(num_lines):
            y = i * step
            color = (0, 255, 0) if i % 2 == 0 else (255, 0, 0)
            cv2.line(img_left_copy, (0, y), (img_left.shape[1], y), color, 1)
            cv2.line(img_right_copy, (0, y), (img_right.shape[1], y), color, 1)
        
        return img_left_copy, img_right_copy
    
    while True:
        # Get trackbar values
        minDisparity = cv2.getTrackbarPos('minDisparity', window_name)
        numDisparities = cv2.getTrackbarPos('numDisparities', window_name) * 16
        if numDisparities == 0:
            numDisparities = 16
        
        blockSize = cv2.getTrackbarPos('blockSize', window_name)
        if blockSize % 2 == 0:
            blockSize += 1
        if blockSize < 5:
            blockSize = 5
        
        P1_mult = cv2.getTrackbarPos('P1_multiplier', window_name)
        P2_mult = cv2.getTrackbarPos('P2_multiplier', window_name)
        disp12MaxDiff = cv2.getTrackbarPos('disp12MaxDiff', window_name)
        uniquenessRatio = cv2.getTrackbarPos('uniquenessRatio', window_name)
        speckleWindowSize = cv2.getTrackbarPos('speckleWindowSize', window_name)
        speckleRange = cv2.getTrackbarPos('speckleRange', window_name)
        preFilterCap = cv2.getTrackbarPos('preFilterCap', window_name)
        
        wls_lambda = cv2.getTrackbarPos('WLS_lambda', window_name)
        wls_sigma = cv2.getTrackbarPos('WLS_sigma', window_name) / 10.0
        
        # Create SGBM object with current parameters
        stereo_left = cv2.StereoSGBM_create(
            minDisparity=minDisparity,
            numDisparities=numDisparities,
            blockSize=blockSize,
            P1=P1_mult * 3 * blockSize**2,
            P2=P2_mult * 3 * blockSize**2,
            disp12MaxDiff=disp12MaxDiff,
            uniquenessRatio=uniquenessRatio,
            speckleWindowSize=speckleWindowSize,
            speckleRange=speckleRange,
            preFilterCap=preFilterCap
        )
        
        # Compute disparity
        left_gray = cv2.cvtColor(left_resize, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_resize, cv2.COLOR_BGR2GRAY)
        
        disparity_left = stereo_left.compute(left_gray, right_gray)
        
        # Apply WLS filter if enabled
        if use_wls_filter:
            # Create right matcher for WLS
            stereo_right = cv2.ximgproc.createRightMatcher(stereo_left)
            disparity_right = stereo_right.compute(right_gray, left_gray)
            
            # Create WLS filter
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(stereo_left)
            wls_filter.setLambda(wls_lambda)
            wls_filter.setSigmaColor(wls_sigma)
            
            # Apply filter
            disparity = wls_filter.filter(disparity_left, left_gray, None, disparity_right)
        else:
            disparity = disparity_left
        
        # Normalize for visualization
        disp = disparity.astype(np.float32) / 16.0
        disp[disp < 0] = 0

        disp_vis = (disp / numDisparities * 255).astype(np.uint8)
        disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

        # Show epipolar lines if enabled
        if show_epipolar:
            left_display, right_display = draw_epipolar_lines(left_resize, right_resize)
            concat = np.hstack((left_display, right_display))
        else:
            # Create display with side-by-side view
            concat = np.hstack((left_resize, disp_color))
        
        # Add parameter text
        params_text = [
            f'minDisparity: {minDisparity}',
            f'numDisparities: {numDisparities}',
            f'blockSize: {blockSize}',
            f'P1: {P1_mult * 3 * blockSize**2}',
            f'P2: {P2_mult * 3 * blockSize**2}',
            f'disp12MaxDiff: {disp12MaxDiff}',
            f'uniquenessRatio: {uniquenessRatio}',
            f'speckleWindowSize: {speckleWindowSize}',
            f'speckleRange: {speckleRange}',
            f'preFilterCap: {preFilterCap}',
            f'WLS Filter: {"ON" if use_wls_filter else "OFF"}',
            f'WLS Lambda: {wls_lambda}',
            f'WLS Sigma: {wls_sigma:.1f}',
            f'Epipolar: {"ON" if show_epipolar else "OFF"}'
        ]
        
        y_offset = 30
        for i, text in enumerate(params_text):
            cv2.putText(concat, text, (10, y_offset + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        cv2.imshow(window_name, concat)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('e'):
            show_epipolar = not show_epipolar
            print(f"Epipolar lines: {'ON' if show_epipolar else 'OFF'}")
        elif key == ord('w'):
            use_wls_filter = not use_wls_filter
            print(f"WLS filter: {'ON' if use_wls_filter else 'OFF'}")
        elif key == ord('n'):
            # Next frame
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                left_frame = frame[:, :width//2]
                right_frame = frame[:, width//2:]
                left_rectified = cv2.remap(left_frame, map1_left, map2_left, cv2.INTER_LINEAR)
                right_rectified = cv2.remap(right_frame, map1_right, map2_right, cv2.INTER_LINEAR)
                left_resize = cv2.resize(left_rectified, (640, 360))
                right_resize = cv2.resize(right_rectified, (640, 360))
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                print("Video ended, looping back to start")
        elif key == ord('s'):
            # Save parameters
            params = {
                'minDisparity': minDisparity,
                'numDisparities': numDisparities,
                'blockSize': blockSize,
                'P1': P1_mult * 3 * blockSize**2,
                'P2': P2_mult * 3 * blockSize**2,
                'disp12MaxDiff': disp12MaxDiff,
                'uniquenessRatio': uniquenessRatio,
                'speckleWindowSize': speckleWindowSize,
                'speckleRange': speckleRange,
                'preFilterCap': preFilterCap,
                'use_wls_filter': use_wls_filter,
                'wls_lambda': wls_lambda,
                'wls_sigma': wls_sigma
            }
            print("\n=== SGBM Parameters ===")
            for key, value in params.items():
                print(f"{key}: {value}")
            print("======================\n")
    
    cap.release()
    cv2.destroyAllWindows()

# Run the tuning function
tune_sgbm_parameters('input_video.mp4', 'stereo_config.yaml')