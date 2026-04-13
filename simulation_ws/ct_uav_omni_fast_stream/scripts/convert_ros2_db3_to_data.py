#!/usr/bin/env python3

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import cv2
from cv_bridge import CvBridge
import sqlite3
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import numpy as np

def get_rosbag_info(db_path):
    """Get topic information from the ROS2 bag database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get topics
    cursor.execute("SELECT id, name, type FROM topics")
    topics = cursor.fetchall()
    
    conn.close()
    return topics

def is_image_topic(msg_type):
    """Check if the message type is an Image type."""
    valid_image_types = [
        'sensor_msgs/msg/Image',
        'sensor_msgs/Image',
    ]
    return msg_type in valid_image_types

def get_windows_compatible_codec(output_video):
    """
    Get the best codec for Windows compatibility based on file extension.
    Returns list of (codec_fourcc, file_extension) to try in order.
    """
    output_path = Path(output_video)
    extension = output_path.suffix.lower()
    
    # Map extensions to codecs (in priority order to try)
    codec_map = {
        '.mp4': [
            ('mp4v', '.mp4'),   # MPEG-4 Part 2 - most compatible
            ('avc1', '.mp4'),   # H.264 - best quality if available
            ('H264', '.mp4'),   # Alternative H.264
            ('X264', '.mp4'),   # Another H.264 variant
        ],
        '.avi': [
            ('MJPG', '.avi'),   # Motion JPEG - very compatible
            ('XVID', '.avi'),   # Xvid codec
            ('DIVX', '.avi'),   # DivX codec
        ],
    }
    
    # Default to MP4 if extension not recognized
    if extension not in codec_map:
        print(f"Warning: Extension '{extension}' not recognized. Using .mp4")
        extension = '.mp4'
    
    codecs = codec_map.get(extension, codec_map['.mp4'])
    
    # Update output path with correct extension
    final_ext = codecs[0][1]
    if output_path.suffix.lower() != final_ext:
        output_video = str(output_path.with_suffix(final_ext))
    
    return codecs, output_video

def convert_bag_to_video(bag_path, topic_name, output_video, fps=30, codec=None):
    """
    Convert ROS2 bag image messages to video.
    
    Args:
        bag_path: Path to the ROS2 bag directory (containing db3 file)
        topic_name: Name of the image topic to extract
        output_video: Output video file path (e.g., 'output.mp4')
        fps: Frames per second for output video
        codec: Video codec (if None, auto-select for Windows compatibility)
    """
    
    # Initialize CV Bridge
    bridge = CvBridge()
    
    # Find the db3 file
    bag_path = Path(bag_path)
    if bag_path.is_file() and bag_path.suffix == '.db3':
        db_file = bag_path
    else:
        # Look for db3 file in directory
        db_files = list(bag_path.glob('*.db3'))
        if not db_files:
            raise FileNotFoundError(f"No .db3 file found in {bag_path}")
        db_file = db_files[0]
    
    print(f"Reading from: {db_file}")
    
    # Connect to the database
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    
    # Get topic id and type
    cursor.execute("SELECT id, type FROM topics WHERE name=?", (topic_name,))
    result = cursor.fetchone()
    
    if not result:
        # Show available topics
        cursor.execute("SELECT name, type FROM topics")
        available = cursor.fetchall()
        conn.close()
        print("\nAvailable topics:")
        for name, msg_type in available:
            is_image = " [IMAGE]" if is_image_topic(msg_type) else ""
            print(f"  {name} ({msg_type}){is_image}")
        raise ValueError(f"\nTopic '{topic_name}' not found.")
    
    topic_id, msg_type = result
    print(f"Topic: {topic_name}")
    print(f"Message type: {msg_type}")
    
    # Verify this is an Image topic
    if not is_image_topic(msg_type):
        conn.close()
        raise ValueError(
            f"Error: Topic '{topic_name}' has type '{msg_type}' which is not an Image type.\n"
            f"This script only works with sensor_msgs/msg/Image topics.\n"
            f"Use --list-topics to see available image topics."
        )
    
    print("✓ Verified: Topic contains Image messages")
    
    # Get message type class
    msg_class = get_message(msg_type)
    
    # Get all messages for this topic
    cursor.execute("""
        SELECT data FROM messages 
        WHERE topic_id=? 
        ORDER BY timestamp
    """, (topic_id,))
    
    messages = cursor.fetchall()
    conn.close()
    
    if not messages:
        raise ValueError(f"No messages found for topic {topic_name}")
    
    print(f"Found {len(messages)} messages")
    
    # Determine codec and ensure Windows compatibility
    if codec is None:
        codec_options, output_video = get_windows_compatible_codec(output_video)
        print(f"Will try codecs in order: {[c[0] for c in codec_options]}")
    else:
        codec_options = [(codec, Path(output_video).suffix)]
        print(f"Using user-specified codec: {codec}")
    
    print(f"Output file: {output_video}")
    
    # Initialize video writer
    video_writer = None
    frame_count = 0
    failed_count = 0
    duplicate_count = 0
    last_valid_frame = None
    start_time = time.time()
    codec_used = None
    first_frame_processed = False
    
    # Process messages with progress bar
    with tqdm(total=len(messages), desc="Converting to video", unit="frame") as pbar:
        for idx, (msg_data,) in enumerate(messages):
            try:
                # Deserialize message
                msg = deserialize_message(msg_data, msg_class)
                
                # Convert ROS image to OpenCV format
                try:
                    cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
                except Exception as e:
                    # Try alternative encoding if bgr8 fails
                    try:
                        cv_image = bridge.imgmsg_to_cv2(msg)
                        # Convert to BGR if needed
                        if len(cv_image.shape) == 2:  # Grayscale
                            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
                        elif cv_image.shape[2] == 4:  # RGBA
                            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGBA2BGR)
                    except Exception as e2:
                        tqdm.write(f"Warning: Failed to convert message {idx}: {e2}")
                        failed_count += 1
                        pbar.update(1)
                        continue
                
                # Validate frame
                if cv_image is None or cv_image.size == 0:
                    tqdm.write(f"Warning: Empty frame at message {idx}")
                    failed_count += 1
                    pbar.update(1)
                    continue
                
                # Check if frame dimensions are valid
                if len(cv_image.shape) < 2 or cv_image.shape[0] == 0 or cv_image.shape[1] == 0:
                    tqdm.write(f"Warning: Invalid frame dimensions at message {idx}")
                    failed_count += 1
                    pbar.update(1)
                    continue
                
                # Initialize video writer with first valid frame dimensions
                if video_writer is None and not first_frame_processed:
                    height, width = cv_image.shape[:2]
                    
                    # Try each codec until one works
                    for codec_attempt, _ in codec_options:
                        try:
                            fourcc = cv2.VideoWriter_fourcc(*codec_attempt)
                            video_writer = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
                            
                            if video_writer.isOpened():
                                codec_used = codec_attempt
                                tqdm.write(f"✓ Video writer initialized successfully")
                                tqdm.write(f"  Codec: {codec_used}")
                                tqdm.write(f"  Dimensions: {width}x{height}")
                                tqdm.write(f"  FPS: {fps}")
                                break
                            else:
                                video_writer.release()
                                video_writer = None
                        except Exception as e:
                            tqdm.write(f"  Failed to use codec '{codec_attempt}': {e}")
                            video_writer = None
                    
                    first_frame_processed = True
                    
                    if video_writer is None or not video_writer.isOpened():
                        raise RuntimeError(
                            f"Failed to initialize video writer with any available codec.\n"
                            f"Tried: {[c[0] for c in codec_options]}\n"
                            f"Please try:\n"
                            f"  1. Installing ffmpeg: 'sudo apt install ffmpeg' (Linux) or download from ffmpeg.org (Windows)\n"
                            f"  2. Reinstalling opencv: 'pip install --upgrade opencv-python'\n"
                            f"  3. Using .avi format: '-o output.avi'"
                        )
                
                # Handle stuck frames (duplicate detection)
                if last_valid_frame is not None:
                    # Check if frame is identical to last frame (stuck camera)
                    if np.array_equal(cv_image, last_valid_frame):
                        duplicate_count += 1
                        # Still write it to maintain timing, but track it
                
                # Write frame
                if video_writer is not None:
                    video_writer.write(cv_image)
                    frame_count += 1
                    last_valid_frame = cv_image.copy()
                
            except Exception as e:
                if "Failed to initialize video writer" in str(e):
                    raise  # Re-raise critical errors
                tqdm.write(f"Warning: Unexpected error at message {idx}: {e}")
                failed_count += 1
            
            # Update progress bar
            pbar.update(1)
            
            # Update progress bar postfix with stats
            elapsed = time.time() - start_time
            fps_processing = (idx + 1) / elapsed if elapsed > 0 else 0
            pbar.set_postfix({
                'Processing FPS': f'{fps_processing:.1f}',
                'Written': frame_count,
                'Failed': failed_count
            })
    
    # Release video writer
    if video_writer:
        video_writer.release()
    
    elapsed_time = time.time() - start_time
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"✓ Successfully created video: {output_video}")
    if codec_used:
        print(f"✓ Codec used: {codec_used}")
    print(f"{'='*60}")
    print(f"Total messages processed: {len(messages)}")
    print(f"Frames written: {frame_count}")
    print(f"Failed conversions: {failed_count}")
    if duplicate_count > 0:
        print(f"Duplicate frames detected: {duplicate_count} (camera stuck/frozen)")
    print(f"Success rate: {(frame_count/len(messages)*100):.1f}%")
    print(f"Video duration: {frame_count / fps:.2f} seconds")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Average processing speed: {frame_count / elapsed_time:.1f} FPS")
    print(f"{'='*60}")
    
    if frame_count == 0:
        raise RuntimeError("No frames were successfully written to video!")
    
    if failed_count > len(messages) * 0.5:
        print(f"WARNING: More than 50% of frames failed to convert. Check your bag file.")

def main():
    parser = argparse.ArgumentParser(
        description='Convert ROS2 bag image data to video (Windows compatible)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List topics in bag
  python script.py /path/to/bag --list-topics
  
  # Convert with auto codec selection (recommended for Windows)
  python script.py /path/to/bag -t /camera/image_raw -o output.mp4
  
  # Custom FPS
  python script.py /path/to/bag -t /camera/image_raw -o output.mp4 -f 60
        """
    )
    parser.add_argument('bag_path', help='Path to ROS2 bag directory or .db3 file')
    parser.add_argument('-t', '--topic', help='Image topic name (required unless --list-topics)')
    parser.add_argument('-o', '--output', default='output.mp4', help='Output video file (default: output.mp4)')
    parser.add_argument('-f', '--fps', type=int, default=30, help='Output video FPS (default: 30)')
    parser.add_argument('-c', '--codec', help='Video codec fourcc (default: auto-select for Windows compatibility)')
    parser.add_argument('--list-topics', action='store_true', help='List available topics and exit')
    
    args = parser.parse_args()
    
    # List topics if requested
    if args.list_topics:
        bag_path = Path(args.bag_path)
        if bag_path.is_file() and bag_path.suffix == '.db3':
            db_file = bag_path
        else:
            db_files = list(bag_path.glob('*.db3'))
            if not db_files:
                print(f"No .db3 file found in {bag_path}")
                return 1
            db_file = db_files[0]
        
        topics = get_rosbag_info(str(db_file))
        print("\nAvailable topics:")
        print(f"{'='*60}")
        for topic_id, name, msg_type in topics:
            is_image = " ✓ [IMAGE]" if is_image_topic(msg_type) else ""
            print(f"{name}")
            print(f"  Type: {msg_type}{is_image}")
        print(f"{'='*60}")
        return 0
    
    # Require topic if not listing
    if not args.topic:
        parser.error("the following arguments are required: -t/--topic (unless using --list-topics)")
    
    # Convert bag to video
    try:
        convert_bag_to_video(args.bag_path, args.topic, args.output, args.fps, args.codec)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())