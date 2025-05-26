"""
SoccerNet Goal Clips Downloader
This script downloads goal event clips from the SoccerNet dataset.
Before using this script:
1. Fill out the NDA form at https://www.soccer-net.org/data
2. You'll receive credentials to download the videos
3. Make sure you have sufficient storage (several GB)
"""

import os
import cv2
import numpy as np
import json
import random
from tqdm import tqdm
from SoccerNet.Downloader import SoccerNetDownloader

# Configuration
SOCCERNET_PATH = "path/to/soccernet"  # Change this to your desired location
OUTPUT_PATH = "goal_clips"            # Folder to save goal clips
NUM_GOALS = 50                        # Number of goal clips to download
CLIP_DURATION = 10                    # Duration of each clip in seconds (5 before + 5 after goal)
PASSWORD = None                       # If you have the password, enter it here

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_PATH, exist_ok=True)

def main():
    print("Setting up SoccerNet downloader...")
    downloader = SoccerNetDownloader(LocalDirectory=SOCCERNET_PATH)
    
    # If you already have the password from the NDA
    if PASSWORD:
        downloader.password = PASSWORD
    else:
        # Prompt for password if not provided
        downloader.password = input("Enter the password received after completing the NDA: ")
    
    print("Downloading SoccerNet annotation labels...")
    # Download action labels
    downloader.downloadGames(files=["Labels-v2.json"], split=["train", "valid", "test"])
    
    # Find all games with goal events
    games_with_goals = find_games_with_goals(SOCCERNET_PATH)
    print(f"Found {len(games_with_goals)} games with goals")
    
    if not games_with_goals:
        print("No games with goals found. Make sure you've downloaded the labels correctly.")
        return
    
    # Select random games until we have enough goals
    random.shuffle(games_with_goals)
    
    # Download videos for selected games with goals
    downloaded_goals = 0
    for game_path, goal_times in games_with_goals:
        # If we already have enough goals, break
        if downloaded_goals >= NUM_GOALS:
            break
            
        # Download the videos for this game
        game_relative_path = os.path.relpath(game_path, SOCCERNET_PATH)
        print(f"Downloading videos for game: {game_relative_path}")
        
        try:
            # Download low-resolution videos (224p) to save space
            downloader.downloadGames(
                files=["1_224p.mkv", "2_224p.mkv"], 
                game=game_relative_path
            )
            
            # Extract clips for each goal in this game
            for half, time_in_half in goal_times:
                if downloaded_goals >= NUM_GOALS:
                    break
                    
                # Determine video file
                video_file = os.path.join(game_path, f"{half}_224p.mkv")
                if not os.path.exists(video_file):
                    print(f"Video file not found: {video_file}")
                    continue
                
                # Extract the clip
                output_file = os.path.join(OUTPUT_PATH, f"goal_{downloaded_goals+1}.mp4")
                success = extract_clip(video_file, time_in_half, output_file)
                
                if success:
                    downloaded_goals += 1
                    print(f"Downloaded goal clip {downloaded_goals}/{NUM_GOALS}")
        
        except Exception as e:
            print(f"Error downloading game {game_relative_path}: {str(e)}")
            continue
    
    print(f"Successfully downloaded {downloaded_goals} goal clips to {OUTPUT_PATH}")

def find_games_with_goals(soccernet_path):
    """Find all games that have goal events and their timestamps"""
    games_with_goals = []
    
    # Walk through the SoccerNet directory structure
    for championship in os.listdir(soccernet_path):
        championship_path = os.path.join(soccernet_path, championship)
        if not os.path.isdir(championship_path):
            continue
            
        for season in os.listdir(championship_path):
            season_path = os.path.join(championship_path, season)
            if not os.path.isdir(season_path):
                continue
                
            for game in os.listdir(season_path):
                game_path = os.path.join(season_path, game)
                if not os.path.isdir(game_path):
                    continue
                
                # Check if Labels-v2.json exists
                labels_file = os.path.join(game_path, "Labels-v2.json")
                if not os.path.exists(labels_file):
                    continue
                
                # Parse the labels file to find goal events
                goal_times = []
                with open(labels_file, 'r') as f:
                    try:
                        annotations = json.load(f)
                        
                        for annotation in annotations["annotations"]:
                            if annotation["label"] == "Goal":
                                # Get the half (1 or 2) and time in the half
                                game_time = annotation["gameTime"]
                                half = 1 if game_time.startswith("1") else 2
                                
                                # Parse the time in the half (in seconds)
                                time_str = game_time.split(" - ")[1]
                                minutes, seconds = map(int, time_str.split(":"))
                                time_in_seconds = minutes * 60 + seconds
                                
                                goal_times.append((half, time_in_seconds))
                    except json.JSONDecodeError:
                        print(f"Error parsing labels file: {labels_file}")
                        continue
                
                if goal_times:
                    games_with_goals.append((game_path, goal_times))
    
    return games_with_goals

def extract_clip(video_file, time_in_seconds, output_file):
    """Extract a clip from a video file around the specified time"""
    try:
        # Calculate start and end times for the clip
        start_time = max(0, time_in_seconds - CLIP_DURATION // 2)
        end_time = time_in_seconds + CLIP_DURATION // 2
        
        # Use OpenCV to extract the clip
        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            print(f"Could not open video file: {video_file}")
            return False
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate frame numbers
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        # Set up video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
        
        # Set the position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        # Read and write frames
        current_frame = start_frame
        while current_frame < end_frame:
            ret, frame = cap.read()
            if not ret:
                break
            
            out.write(frame)
            current_frame += 1
        
        # Release resources
        cap.release()
        out.release()
        
        # Check if we successfully created the clip
        if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
            return True
        else:
            print(f"Failed to create clip: {output_file}")
            return False
            
    except Exception as e:
        print(f"Error extracting clip: {str(e)}")
        return False

if __name__ == "__main__":
    main()