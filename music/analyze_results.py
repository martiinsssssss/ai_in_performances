import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

def analyze_data(data_folder="collected_data_osc"):
    """
    Reads session data, calculates metrics, generates comparative and 3D plots,
    and saves a summary of the metrics to a CSV file.

    Args:
        data_folder (str): The folder containing the collected CSV files.
    """
    session_metrics = []
    trajectory_data_for_plots = []

    for filename in os.listdir(data_folder):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_folder, filename)
            session_prefix = '_'.join(filename.split('_')[:-1]) 

            print(f"Processing file: {filename}")
            df = pd.read_csv(filepath)

            # Calculate Metrics per Session
            avg_latency_ms = df['frame_processing_time_ms'].mean()
            num_frames = df['global_timestamp'].nunique()
            session_duration = df['global_timestamp'].max() - df['global_timestamp'].min()
            fps = num_frames / session_duration if session_duration > 0 else 0

            detected_hands_df = df[(df['hand_id'] != -1) & (df['hand_confidence_score'] > 0)]
            avg_confidence = detected_hands_df['hand_confidence_score'].mean() if not detected_hands_df.empty else 0

            no_detection_frames = df[df['hand_id'] == -1].shape[0]
            total_frames = df.shape[0]
            no_detection_percentage = (no_detection_frames / total_frames) * 100 if total_frames > 0 else 0

            target_landmark_id = 0 # Wrist landmark ID
            hand_positions = df[(df['landmark_id'] == target_landmark_id) & (df['hand_id'] == 0)].copy() 
            
            frame_w = df['frame_width_px'].iloc[0] if not df.empty else 1
            frame_h = df['frame_height_px'].iloc[0] if not df.empty else 1
            
            hand_x_pixels = hand_positions['x_normalized'] * frame_w 
            hand_y_pixels = hand_positions['y_normalized'] * frame_h 

            jitter_x = hand_x_pixels.std() if not hand_x_pixels.empty else np.nan
            jitter_y = hand_y_pixels.std() if not hand_y_pixels.empty else np.nan
            
            total_jitter_pixels = np.sqrt(jitter_x**2 + jitter_y**2) if not (np.isnan(jitter_x) or np.isnan(jitter_y)) else np.nan

            session_metrics.append({
                'Session': session_prefix,
                'Avg Latency (ms)': avg_latency_ms,
                'FPS': fps,
                'Avg Confidence': avg_confidence,
                'No Detection %': no_detection_percentage,
                'Jitter X (px)': jitter_x,
                'Jitter Y (px)': jitter_y,
                'Total Jitter (px)': total_jitter_pixels 
            })

            if not hand_positions.empty:
                hand_positions['x_pixel'] = hand_positions['x_normalized'] * frame_w
                hand_positions['y_pixel'] = hand_positions['y_normalized'] * frame_h
                hand_positions['Session'] = session_prefix
                trajectory_data_for_plots.append(hand_positions)

    if not session_metrics:
        print("No CSV files found in the specified folder. Ensure 'collect_data_reliable.py' was executed first.")
        return

    metrics_df = pd.DataFrame(session_metrics)
    print("\n--- Session Metrics Summary ---")
    print(metrics_df)

    plots_dir = "analysis_plots"
    os.makedirs(plots_dir, exist_ok=True)

    summary_output_filename = os.path.join(plots_dir, "session_metrics_summary.csv")
    metrics_df.to_csv(summary_output_filename, index=False)
    print(f"\nSummary metrics saved to: {summary_output_filename}")

    # Plot 1: Average Latency
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Session', y='Avg Latency (ms)', data=metrics_df)
    plt.title('Average Hand Detection Latency per Scenario')
    plt.ylabel('Latency (ms)')
    plt.xlabel('Test Scenario')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'avg_latency_plot.png'))
    plt.show()

    # Plot 2: Frames Per Second
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Session', y='FPS', data=metrics_df)
    plt.title('Frames Per Second (FPS) per Scenario')
    plt.ylabel('FPS')
    plt.xlabel('Test Scenario')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'fps_plot.png'))
    plt.show()

    # Plot 3: Average Confidence Score
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Session', y='Avg Confidence', data=metrics_df)
    plt.title('Average Hand Detection Confidence Score per Scenario')
    plt.ylabel('Average Confidence (0-1)')
    plt.xlabel('Test Scenario')
    plt.ylim(0, 1)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'avg_confidence_plot.png'))
    plt.show()

    # Plot 4: Percentage of Frames with No Detection
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Session', y='No Detection %', data=metrics_df)
    plt.title('Percentage of Frames with No Hand Detection per Scenario')
    plt.ylabel('% of Frames')
    plt.xlabel('Test Scenario')
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'no_detection_percentage_plot.png'))
    plt.show()

    # Plot 5: Total Jitter (Stability)
    if 'Total Jitter (px)' in metrics_df.columns and not metrics_df['Total Jitter (px)'].isnull().all():
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Session', y='Total Jitter (px)', data=metrics_df.dropna(subset=['Total Jitter (px)']))
        plt.title(f'Total Jitter (Stability) for Landmark {target_landmark_id} (Wrist) per Scenario')
        plt.ylabel('Standard Deviation (pixels)')
        plt.xlabel('Test Scenario')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, 'total_jitter_plot.png'))
        plt.show()
    else:
        print("No Total Jitter data to plot.")

    # 3D Trajectory Plots for each session
    if trajectory_data_for_plots:
        for df_traj in trajectory_data_for_plots:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            ax.plot(df_traj['x_pixel'], df_traj['y_pixel'], df_traj['z_relative'], marker='o', markersize=2, linestyle='-', alpha=0.7)
            
            ax.set_title(f'3D Trajectory of Wrist (Landmark {target_landmark_id}) for Session: {df_traj["Session"].iloc[0]}')
            ax.set_xlabel('X Position (pixels)')
            ax.set_ylabel('Y Position (pixels)')
            ax.set_zlabel('Relative Z (depth)')
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f'3d_trajectory_{df_traj["Session"].iloc[0]}.png'))
            plt.show()
    else:
        print("No trajectory data available to generate 3D plots.")

    print(f"\nAnalysis complete. Plots have been saved to the '{plots_dir}' folder.")

if __name__ == "__main__":
    analyze_data()