#!/usr/bin/env python3

import rclpy
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
import cv2
import numpy as np
import sqlite3
import argparse
from pathlib import Path
from tqdm import tqdm
import csv
import pandas as pd
from collections import defaultdict

def decompress_image(compressed_msg):
    """Decompress CompressedImage message to OpenCV image."""
    np_arr = np.frombuffer(compressed_msg.data, np.uint8)
    cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    return cv_image

def extract_messages_from_bag(db_path, topics):
    """
    Extract all messages from specified topics with timestamps.
    Returns: dict {topic_name: [(timestamp, msg_data), ...]}
    """
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Get topic IDs
    topic_ids = {}
    topic_types = {}
    for topic in topics:
        cursor.execute("SELECT id, type FROM topics WHERE name=?", (topic,))
        result = cursor.fetchone()
        if result:
            topic_ids[topic] = result[0]
            topic_types[topic] = result[1]
    
    if not topic_ids:
        conn.close()
        raise ValueError(f"None of the specified topics found in bag")
    
    print(f"Found topics:")
    for topic, topic_id in topic_ids.items():
        print(f"  {topic} (ID: {topic_id}, Type: {topic_types[topic]})")
    
    # Extract messages for each topic
    messages_by_topic = defaultdict(list)
    
    for topic, topic_id in topic_ids.items():
        cursor.execute("""
            SELECT timestamp, data 
            FROM messages 
            WHERE topic_id=? 
            ORDER BY timestamp
        """, (topic_id,))
        
        rows = cursor.fetchall()
        msg_class = get_message(topic_types[topic])
        
        for timestamp, msg_data in rows:
            msg = deserialize_message(msg_data, msg_class)
            messages_by_topic[topic].append((timestamp, msg))
        
        print(f"  Loaded {len(rows)} messages from {topic}")
    
    conn.close()
    return messages_by_topic, topic_types

def synchronize_messages(messages_by_topic, time_tolerance_ns=50_000_000):
    """
    Synchronize messages from multiple topics based on timestamp.
    time_tolerance_ns: Maximum time difference in nanoseconds (default: 50ms)
    
    Returns: List of synchronized message groups
    """
    topics = list(messages_by_topic.keys())
    
    # Create indices for each topic
    indices = {topic: 0 for topic in topics}
    synchronized = []
    
    # Find the topic with the minimum number of messages
    min_messages = min(len(messages_by_topic[topic]) for topic in topics)
    
    print(f"\nSynchronizing {len(topics)} topics...")
    print(f"Time tolerance: {time_tolerance_ns / 1_000_000:.1f} ms")
    
    with tqdm(total=min_messages, desc="Synchronizing messages") as pbar:
        while all(indices[topic] < len(messages_by_topic[topic]) for topic in topics):
            # Get current timestamps for all topics
            current_timestamps = {
                topic: messages_by_topic[topic][indices[topic]][0]
                for topic in topics
            }
            
            # Find the reference timestamp (median or earliest)
            ref_timestamp = sorted(current_timestamps.values())[len(topics) // 2]
            
            # Check if all timestamps are within tolerance
            time_diffs = [abs(ts - ref_timestamp) for ts in current_timestamps.values()]
            max_diff = max(time_diffs)
            
            if max_diff <= time_tolerance_ns:
                # All messages are synchronized
                sync_group = {
                    'timestamp': ref_timestamp,
                    'messages': {}
                }
                
                for topic in topics:
                    sync_group['messages'][topic] = messages_by_topic[topic][indices[topic]][1]
                    indices[topic] += 1
                
                synchronized.append(sync_group)
                pbar.update(1)
            else:
                # Advance the topic with the earliest timestamp
                earliest_topic = min(current_timestamps, key=current_timestamps.get)
                indices[earliest_topic] += 1
    
    print(f"✓ Synchronized {len(synchronized)} message groups")
    return synchronized

def save_images(synchronized_data, output_dir, topic_names):
    """Save all images to disk."""
    images_dir = output_dir / 'images'
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for each topic
    topic_dirs = {}
    for topic in topic_names:
        # Clean topic name for directory
        dir_name = topic.replace('/', '_').strip('_')
        topic_dir = images_dir / dir_name
        topic_dir.mkdir(exist_ok=True)
        topic_dirs[topic] = topic_dir
    
    print(f"\nSaving images to {images_dir}...")
    
    image_paths = defaultdict(list)
    
    for idx, sync_group in enumerate(tqdm(synchronized_data, desc="Saving images")):
        timestamp = sync_group['timestamp']
        
        for topic, msg in sync_group['messages'].items():
            # Decompress image
            cv_image = decompress_image(msg)
            
            if cv_image is not None:
                # Save image
                filename = f"{timestamp:019d}.jpg"
                filepath = topic_dirs[topic] / filename
                cv2.imwrite(str(filepath), cv_image)
                image_paths[topic].append(str(filepath.relative_to(output_dir)))
            else:
                image_paths[topic].append(None)
    
    print(f"✓ Saved images for {len(topic_names)} topics")
    return image_paths

def create_videos(synchronized_data, output_dir, topic_configs, fps=30):
    """
    Create videos for specified topics.
    topic_configs: dict {topic_name: {'color': bool, 'name': str}}
    """
    video_dir = output_dir / 'video'
    video_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating videos in {video_dir}...")
    
    video_writers = {}
    video_paths = {}
    frame_counts = defaultdict(int)
    
    # Initialize video writers
    for topic, config in topic_configs.items():
        video_name = config['name']
        video_path = video_dir / f"{video_name}.avi"
        video_paths[topic] = str(video_path)
        video_writers[topic] = None
    
    # Process messages
    for sync_group in tqdm(synchronized_data, desc="Creating videos"):
        for topic, config in topic_configs.items():
            if topic not in sync_group['messages']:
                continue
            
            msg = sync_group['messages'][topic]
            cv_image = decompress_image(msg)
            
            if cv_image is None:
                continue
            
            # Convert to grayscale if needed
            if not config['color']:
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
                cv_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2BGR)
            
            # Initialize video writer on first frame
            if video_writers[topic] is None:
                height, width = cv_image.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'MJPG')
                video_writers[topic] = cv2.VideoWriter(
                    video_paths[topic], fourcc, fps, (width, height)
                )
                
                if not video_writers[topic].isOpened():
                    print(f"Warning: Failed to open video writer for {topic}")
                    continue
            
            # Write frame
            video_writers[topic].write(cv_image)
            frame_counts[topic] += 1
    
    # Release all video writers
    for writer in video_writers.values():
        if writer is not None:
            writer.release()
    
    print(f"✓ Created {len(video_paths)} videos")
    for topic, path in video_paths.items():
        print(f"  {Path(path).name}: {frame_counts[topic]} frames")
    
    return video_paths

def save_csv(synchronized_data, output_dir, image_paths, topic_names):
    """Save synchronized data to CSV file."""
    csv_path = output_dir / 'synchronized_data.csv'
    
    print(f"\nSaving CSV to {csv_path}...")
    
    # Prepare CSV data
    rows = []
    for idx, sync_group in enumerate(synchronized_data):
        row = {'timestamp': sync_group['timestamp']}
        
        for topic in topic_names:
            # Clean topic name for column
            col_name = topic.replace('/segmentation_record_node/', '').replace('/compressed', '')
            
            if idx < len(image_paths[topic]) and image_paths[topic][idx]:
                row[col_name] = image_paths[topic][idx]
            else:
                row[col_name] = None
        
        rows.append(row)
    
    # Write to CSV
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    print(f"✓ Saved CSV with {len(rows)} rows")
    print(f"  Columns: {', '.join(df.columns)}")
    
    return csv_path

def main():
    parser = argparse.ArgumentParser(
        description='Process ROS2 bag with compressed images',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('bag_path', help='Path to ROS2 bag directory or .db3 file')
    parser.add_argument('-o', '--output', default='data', help='Output directory (default: data)')
    parser.add_argument('-f', '--fps', type=int, default=30, help='Video FPS (default: 30)')
    parser.add_argument('-t', '--tolerance', type=float, default=50.0, 
                       help='Time synchronization tolerance in milliseconds (default: 50ms)')
    
    args = parser.parse_args()
    
    # Find db3 file
    bag_path = Path(args.bag_path)
    if bag_path.is_file() and bag_path.suffix == '.db3':
        db_file = bag_path
    else:
        db_files = list(bag_path.glob('*.db3'))
        if not db_files:
            print(f"Error: No .db3 file found in {bag_path}")
            return 1
        db_file = db_files[0]
    
    print(f"Processing bag: {db_file}")
    
    # Define topics
    topics = [
        '/segmentation_record_node/rgb_cam0/compressed',
        '/segmentation_record_node/rgb_cam1/compressed',
        '/segmentation_record_node/segmentation_cam0/compressed',
        '/segmentation_record_node/segmentation_cam1/compressed',
    ]
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir.absolute()}")
    
    try:
        # Step 1: Extract messages
        print("\n" + "="*60)
        print("STEP 1: Extracting messages from bag")
        print("="*60)
        messages_by_topic, topic_types = extract_messages_from_bag(db_file, topics)
        
        # Step 2: Synchronize messages
        print("\n" + "="*60)
        print("STEP 2: Synchronizing messages")
        print("="*60)
        time_tolerance_ns = int(args.tolerance * 1_000_000)  # Convert ms to ns
        synchronized_data = synchronize_messages(messages_by_topic, time_tolerance_ns)
        
        if not synchronized_data:
            print("Error: No synchronized messages found!")
            return 1
        
        # Step 3: Save images
        print("\n" + "="*60)
        print("STEP 3: Saving images")
        print("="*60)
        image_paths = save_images(synchronized_data, output_dir, topics)
        
        # Step 4: Create videos
        print("\n" + "="*60)
        print("STEP 4: Creating videos")
        print("="*60)
        video_configs = {
            '/segmentation_record_node/rgb_cam0/compressed': {
                'color': True, 
                'name': 'rgb_cam0_color'
            },
            '/segmentation_record_node/rgb_cam1/compressed': {
                'color': True, 
                'name': 'rgb_cam1_color'
            },
            '/segmentation_record_node/segmentation_cam0/compressed': {
                'color': False, 
                'name': 'segmentation_cam0_gray'
            },
            '/segmentation_record_node/segmentation_cam1/compressed': {
                'color': False, 
                'name': 'segmentation_cam1_gray'
            },
        }
        video_paths = create_videos(synchronized_data, output_dir, video_configs, args.fps)
        
        # Step 5: Save CSV
        print("\n" + "="*60)
        print("STEP 5: Saving CSV")
        print("="*60)
        csv_path = save_csv(synchronized_data, output_dir, image_paths, topics)
        
        # Summary
        print("\n" + "="*60)
        print("✓ PROCESSING COMPLETE")
        print("="*60)
        print(f"Output directory: {output_dir.absolute()}")
        print(f"\nContents:")
        print(f"  📁 images/          - {len(synchronized_data)} synchronized image sets")
        print(f"  📁 video/           - 4 video files")
        print(f"  📄 synchronized_data.csv - Metadata with {len(synchronized_data)} rows")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())