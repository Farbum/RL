import cv2
from moviepy.editor import VideoFileClip, concatenate_videoclips, TextClip, CompositeVideoClip
import glob
import os
import re


def add_text(cv2_img, text, position=(50, 50), font_scale=1, font_color=(255, 255, 255)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(cv2_img, text, position, font, font_scale, font_color, 2, cv2.LINE_AA)
    return cv2_img

def make_text_lambda(text):
    return lambda image: add_text(image, text, position=(50, 50))


# These couple of functions to sort file alphabetically with numbers
def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



root = "/home/had/Python works/RL - hugging face/PPO/videos"
exp  = "LunarLander-v2__Lunar_Lander__1__1701462943"
vid_path = os.path.join(root, exp)

# List of video file paths
video_files = glob.glob(os.path.join(vid_path, "*.mp4"))
video_files.sort(key=natural_keys)
text_list = [f'{x}th Episode' for x in [0,50,100,150,200,250,300]]  # Corresponding text for each video
processed_clips = []

for i, (filename, text_for_clip) in enumerate(zip(video_files, text_list)):
    # Load video
    clip = VideoFileClip(filename)
    
    # Accelerate video by a factor X
    factor_x = 2  # Example factor
    accelerated_clip = clip.speedx(factor_x)

    processed_clip = accelerated_clip.fl_image(make_text_lambda(text_for_clip))
    processed_clips.append(processed_clip)

# Concatenate and write the final video
final_clip = concatenate_videoclips(processed_clips)
# Write the result to a file
final_clip.write_videofile(os.path.join(vid_path, "movie.mp4"))