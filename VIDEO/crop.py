from moviepy.editor import VideoFileClip
from moviepy.video.fx.all import crop, mirror_x
# names = ['1_left_front', '2_left_front', '3_left_front', '4_left_front', '5_left_front']
# names = ['1_side', '2_side', '3_side']
name = 'side'

for idx in range(1, 4):
    video = VideoFileClip(f"crop/{str(idx)}_{name}.mp4")
    # front
    # output = crop(video, x1=600, y1=100, width=800, height=800) 
    # left front
    # output = crop(video, x1=400, y1=150, width=900, height=900) 
    output = mirror_x(video)

    output.write_videofile(f"{str(idx+5)}_{name}.mp4", fps=30, temp_audiofile="temp-audio.m4a", remove_temp=True, codec="libx264", audio_codec="aac")
    print('ok')