"""
Old Method (2000): Computer Vision System for Tracking Players in Sports Games
Based on: Perš & Kovačič (2000)

Methods:
1. Motion Detection - Frame subtraction
2. Template Tracking - 14 Walsh-like templates
3. Color Tracking - RGB similarity
4. Radial Distortion Correction
"""

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.spatial.distance import euclidean
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
video_path = os.path.join(BASE_DIR, "old.mp4")
cap = cv2.VideoCapture(video_path)


class OldMethodTracker:
    """
    Implementation of 2000 paper tracking methods
    """
    
    def __init__(self, video_path):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.reference_frame = None
        self.player_colors = []
        self.trajectories = []
        self.manual_interventions = 0
        
        # Template masks (14 Walsh-like functions)
        self.templates = self._create_templates()
        
    def _create_templates(self):
        """
        Create 14 Walsh-like template functions (Section 4.2)
        Templates are 16x16 binary masks
        """
        templates = []
        size = 16
        
        # Template 1-4: Vertical splits
        for i in range(4):
            template = np.zeros((size, size))
            split = size // (i + 2)
            template[:, :split] = 1
            templates.append(template)
        
        # Template 5-8: Horizontal splits
        for i in range(4):
            template = np.zeros((size, size))
            split = size // (i + 2)
            template[:split, :] = 1
            templates.append(template)
        
        # Template 9-10: Diagonal patterns
        template = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i + j < size:
                    template[i, j] = 1
        templates.append(template)
        
        template = np.zeros((size, size))
        for i in range(size):
            for j in range(size):
                if i > j:
                    template[i, j] = 1
        templates.append(template)
        
        # Template 11-14: Quadrant patterns
        for i in range(4):
            template = np.zeros((size, size))
            if i == 0:
                template[:size//2, :size//2] = 1
            elif i == 1:
                template[:size//2, size//2:] = 1
            elif i == 2:
                template[size//2:, :size//2] = 1
            else:
                template[size//2:, size//2:] = 1
            templates.append(template)
        
        return templates
    
    def set_reference_frame(self, frame):
        """
        Set reference frame (empty court) for motion detection (Section 4.1)
        """
        self.reference_frame = frame.copy()
        
    def motion_detection(self, frame):
        """
        Method A: Motion detection using frame subtraction (Section 4.1)
        
        Returns:
            Blob positions and their centers of gravity
        """
        if self.reference_frame is None:
            raise ValueError("Reference frame not set!")
        
        # Equation (5): D = |R_R - C_R| + |R_G - C_G| + |R_B - C_B|
        diff_R = np.abs(self.reference_frame[:,:,2].astype(float) - frame[:,:,2].astype(float))
        diff_G = np.abs(self.reference_frame[:,:,1].astype(float) - frame[:,:,1].astype(float))
        diff_B = np.abs(self.reference_frame[:,:,0].astype(float) - frame[:,:,0].astype(float))
        
        diff_image = diff_R + diff_G + diff_B
        
        # Threshold
        threshold = 50  # Fixed threshold as mentioned in paper
        binary = (diff_image > threshold).astype(np.uint8) * 255
        
        # Filter to reduce noise
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
        
        # Find blobs
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
        
        # Filter blobs by size (players are 10-15 pixels in diameter as mentioned)
        min_area = 50
        max_area = 500
        
        player_positions = []
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            if min_area < area < max_area:
                player_positions.append(centroids[i])
        
        return player_positions, binary
    
    def template_tracking(self, frame, position):
        """
        Method using template matching (Section 4.2)
        
        Args:
            frame: Current frame
            position: (x, y) center position to extract features
            
        Returns:
            Feature vector (42 dimensions: 14 templates x 3 RGB channels)
        """
        x, y = int(position[0]), int(position[1])
        size = 16
        
        # Extract 16x16 region around position
        x1 = max(0, x - size//2)
        y1 = max(0, y - size//2)
        x2 = min(frame.shape[1], x + size//2)
        y2 = min(frame.shape[0], y + size//2)
        
        region = frame[y1:y2, x1:x2]
        
        # Resize to 16x16 if needed
        if region.shape[0] != size or region.shape[1] != size:
            region = cv2.resize(region, (size, size))
        
        # Extract features for each channel (Equation 6)
        features = []
        for channel in range(3):  # B, G, R
            channel_data = region[:, :, channel].astype(float) / 255.0
            for template in self.templates:
                # F_j = (1/16^2) * sum(I(x,y) * K_j(x,y))
                feature = np.sum(channel_data * template) / (size * size)
                features.append(feature)
        
        return np.array(features)
    
    def calculate_similarity(self, features_current, features_background, features_history):
        """
        Calculate similarity measure (Equation 7)
        
        s = d^2 / (d_GF^2 + d_FH^2)
        where d_GF is distance to background, d_FH is distance to player history
        """
        d_GF = euclidean(features_current, features_background)
        d_FH = euclidean(features_current, features_history)
        
        if d_GF + d_FH == 0:
            return 0
        
        similarity = (d_GF ** 2) / (d_GF ** 2 + d_FH ** 2)
        return similarity
    
    def color_tracking(self, frame, position, player_color, search_radius=30):
        """
        Method B: Color tracking (Section 4.3)
        
        Args:
            frame: Current frame
            position: Previous position (x, y)
            player_color: RGB color of player's dress
            search_radius: Radius to search around previous position
            
        Returns:
            New position of player
        """
        x, y = int(position[0]), int(position[1])
        
        # Define search area
        x1 = max(0, x - search_radius)
        y1 = max(0, y - search_radius)
        x2 = min(frame.shape[1], x + search_radius)
        y2 = min(frame.shape[0], y + search_radius)
        
        search_region = frame[y1:y2, x1:x2]
        
        # Equation (8): S(x,y) = sqrt((C_R - I_R)^2 + (C_G - I_G)^2 + (C_B - I_B)^2)
        diff_R = (player_color[2] - search_region[:,:,2].astype(float)) ** 2
        diff_G = (player_color[1] - search_region[:,:,1].astype(float)) ** 2
        diff_B = (player_color[0] - search_region[:,:,0].astype(float)) ** 2
        
        similarity = np.sqrt(diff_R + diff_G + diff_B)
        
        # Find pixel with minimum distance (most similar)
        min_idx = np.unravel_index(np.argmin(similarity), similarity.shape)
        
        # Convert back to frame coordinates
        new_x = x1 + min_idx[1]
        new_y = y1 + min_idx[0]
        
        return (new_x, new_y)
    
    def correct_radial_distortion(self, point, h, camera_center):
        """
        Correct radial distortion using custom model (Section 3)
        
        Equations (3) and (4) from paper
        r1 = h * ln(sqrt(1 + (R1/h)^2) + R1/h)
        R1 = h * (e^(r1/h) - e^(-r1/h)) / (e^(r1/h) + e^(-r1/h))
        """
        x, y = point
        cx, cy = camera_center
        
        # Calculate r1 (distance from image center)
        r1 = np.sqrt((x - cx)**2 + (y - cy)**2)
        
        if r1 == 0:
            return point
        
        # Apply inverse distortion model (Equation 4)
        exp_term = np.exp(r1 / h)
        R1 = h * (exp_term - 1/exp_term) / (exp_term + 1/exp_term)
        
        # Scale back to image coordinates
        scale = R1 / r1
        corrected_x = cx + (x - cx) * scale
        corrected_y = cy + (y - cy) * scale
        
        return (corrected_x, corrected_y)
    
    def combined_tracking(self, frame, reference_frame, prev_positions, player_colors):
        """
        Method C: Combined color and template tracking
        Best method from Table 1: 14 interventions, 55m noise
        """
        new_positions = []
        
        for i, (prev_pos, color) in enumerate(zip(prev_positions, player_colors)):
            # First, use color tracking for rough position
            color_pos = self.color_tracking(frame, prev_pos, color)
            
            # Then refine with template tracking
            current_features = self.template_tracking(frame, color_pos)
            bg_features = self.template_tracking(reference_frame, color_pos)
            
            # Use history (simplified: just use current as history for demo)
            similarity = self.calculate_similarity(current_features, bg_features, current_features)
            
            # If similarity is low, might need manual intervention
            if similarity < 0.3:
                self.manual_interventions += 1
                print(f" Manual intervention needed for player {i}")
            
            new_positions.append(color_pos)
        
        return new_positions
    
    def process_video(self, method='C', num_players=6):
        """
        Process entire video using specified method
        
        Args:
            method: 'A' (motion), 'B' (color), or 'C' (combined)
            num_players: Number of players to track
        """
        print(" ")
        print(f" OLD METHOD (2000) - Method {method}")
        print(" ")
        
        # Read first frame as reference
        ret, first_frame = self.cap.read()
        if not ret:
            raise ValueError("Cannot read video")
        
        self.set_reference_frame(first_frame)
        
        # Initialize tracking
        all_trajectories = [[] for _ in range(num_players)]
        
        # Define player colors (would be set by operator)
        np.random.seed(42)
        player_colors = [
            np.random.randint(50, 255, 3).tolist() for _ in range(num_players)
        ]
        
        # Get initial positions from first frame
        positions, _ = self.motion_detection(first_frame)
        if len(positions) < num_players:
            # Pad with random positions if needed
            for i in range(num_players - len(positions)):
                positions.append((np.random.rand() * first_frame.shape[1],
                                np.random.rand() * first_frame.shape[0]))
        positions = positions[:num_players]
        
        frame_count = 0
        total_distance = 0  # For noise measurement
        
        print("\nProcessing frames...")
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            if method == 'A':
                # Motion detection only
                new_positions, _ = self.motion_detection(frame)
                if len(new_positions) >= num_players:
                    positions = new_positions[:num_players]
                
            elif method == 'B':
                # Color tracking
                new_positions = []
                for pos, color in zip(positions, player_colors):
                    new_pos = self.color_tracking(frame, pos, color)
                    new_positions.append(new_pos)
                    # Calculate noise (distance moved)
                    dist = euclidean(pos, new_pos)
                    total_distance += dist
                positions = new_positions
                
            else:  # Method C
                # Combined method
                positions = self.combined_tracking(frame, self.reference_frame, 
                                                  positions, player_colors)
            
            # Store trajectories
            for i, pos in enumerate(positions):
                all_trajectories[i].append(pos)
            
            if frame_count % 25 == 0:
                print(f"  Processed {frame_count} frames, "
                      f"Interventions: {self.manual_interventions}")
        
        self.cap.release()
        
        # Print results matching Table 1
        print(" ")
        print(" RESULTS (Table 1 format)")
        print(" ")
        print(f"Method: {method}")
        print(f"Interventions: {self.manual_interventions}")
        print(f"Total Distance: {total_distance:.1f} pixels")
        print(f"Frames Processed: {frame_count}")
        print(f"Average FPS: 25 (PAL standard)")
        
        return all_trajectories
    
    def visualize_trajectories(self, trajectories, output_path='trajectories_old.png'):
        """
        Visualize player trajectories (similar to Figure 6)
        """
        plt.figure(figsize=(14, 8))
        
        colors = plt.cm.rainbow(np.linspace(0, 1, len(trajectories)))
        
        for i, traj in enumerate(trajectories):
            if len(traj) > 0:
                traj = np.array(traj)
                # Apply Gaussian filter as mentioned in paper
                traj_filtered = gaussian_filter(traj, sigma=[2, 0])
                plt.plot(traj_filtered[:, 0], traj_filtered[:, 1], 
                        color=colors[i], linewidth=2, label=f'Player {i+1}',
                        alpha=0.7)
                plt.scatter(traj_filtered[0, 0], traj_filtered[0, 1],
                          color=colors[i], s=100, marker='o', edgecolor='black')
                plt.scatter(traj_filtered[-1, 0], traj_filtered[-1, 1],
                          color=colors[i], s=100, marker='s', edgecolor='black')
        
        plt.xlabel('X Position (pixels)', fontsize=12)
        plt.ylabel('Y Position (pixels)', fontsize=12)
        plt.title('Player Trajectories - Old Method (2000)', fontsize=14, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.gca().invert_yaxis()  # Invert Y axis (image coordinates)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        print(f"\n Trajectories saved to {output_path}")


if __name__ == "__main__":
    # Example usage
    print(" ")
    print(" OLD METHOD (2000) IMPLEMENTATION")
    print(" Based on: Perš & Kovačič (2000)")
    print(" ")
    
    #video_path = "old.mp4"
    
    try:
        tracker = OldMethodTracker(video_path)
        
        # Test all three methods as in Table 1
        for method in ['A', 'B', 'C']:
            tracker = OldMethodTracker(video_path)
            trajectories = tracker.process_video(method=method, num_players=6)
            tracker.visualize_trajectories(trajectories, 
                                          f'trajectories_method_{method}.png')
            print("\n")
        
        print(" ")
        print(" COMPARISON WITH TABLE 1")
        print(" ")
        print("Method A (Motion Detection):")
        print("  Paper: 45 interventions, 80m noise, 0.424 sec/frame")
        print("  Our Implementation: See above results")
        print("\nMethod B (Color Tracking):")
        print("  Paper: 12 interventions, 249m noise, 0.175 sec/frame")
        print("  Our Implementation: See above results")
        print("\nMethod C (Combined - BEST):")
        print("  Paper: 14 interventions, 55m noise, 0.229 sec/frame")
        print("  Our Implementation: See above results")
        
    except FileNotFoundError:
        print(f"\n Error: Video file '{video_path}' not found!")
        print("Please provide old.mp4 video file")
