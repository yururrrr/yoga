import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import argparse
import os
import pandas as pd
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='video folder path')
    parser.add_argument('--save_path', type=str, help='save folder path')

    return parser.parse_args()
    # os.makedirs(args.save_path, exist_ok=True)

# Preprocess input image
def preprocess_image(image):
    image_tensor = F.to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

# Perform inference
def detect_objects(model, frame):
    # Convert the frame to a PyTorch tensor
    image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float()  # Convert to CHW format
    image_tensor /= 255.0  # Normalize to [0, 1]
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions

# Visualize results
def visualize_result(frame, predictions):
    max_score = 0
    
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        xmin, ymin, xmax, ymax = box.tolist()  # Convert tensor to list
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)  # Convert to integers
        if label.item() == 1 and score.item() > 0.5:  # Filter for 'person' class
            if score.item() > max_score:
                max_score = score.item()

                # Calculate the larger square bounding box
                width = xmax - xmin
                height = ymax - ymin
                max_dim = max(width, height)
                center_x = xmin + width // 2
                center_y = ymin + height // 2
                square_x1 = center_x - max_dim // 2
                square_y1 = center_y - max_dim // 2
                square_x2 = square_x1 + max_dim
                square_y2 = square_y1 + max_dim
                frame = cv2.rectangle(frame, (square_x1, square_y1), (square_x2, square_y2), (0, 255, 0), 2)
                
                # Crop the frame to the square bounding box
                cropped_frame = frame[square_y1:square_y2, square_x1:square_x2]
            else:
                cropped_frame = None

    return frame, cropped_frame

def video2class(pose, output_path, v_num):
    
    if pose ==1:    output_frame_path = f'{output_path}mountain pose/{v_num}/'
    elif pose ==2:  output_frame_path = f'{output_path}raised hands pose/{v_num}/'
    elif pose ==3:  output_frame_path = f'{output_path}standing forward bend/{v_num}/'
    elif pose==4:   output_frame_path = f'{output_path}half standing forward bend/{v_num}/'
    elif pose==5:   output_frame_path = f'{output_path}plank/{v_num}/'
    elif pose==6:   output_frame_path = f'{output_path}four-limbed staff pose/{v_num}/'
    elif pose==7:   output_frame_path = f'{output_path}upward facing dog/{v_num}/'
    elif pose==8:   output_frame_path = f'{output_path}downward facing dog/{v_num}/'
    
    os.makedirs(output_frame_path, exist_ok=True)
    return output_frame_path

def main():
    args = parse_args()
    if args.data_path is None or not os.path.exists(args.data_path):
        print("Please provide a valid data path.")
        return
    
    home_path = 'C:/Users/VIPLAB/Desktop/yuru/'
    data_path = os.path.join(home_path, args.data_path)
    video_files = [f for f in os.listdir(data_path) if f.endswith('.MP4')]

    labels=['downward facing dog', 'four-limbed staff pose', 'half standing forward bend', 'mountain pose', 'plank', 'raised hands pose', 'standing forward bend', 'upward facing dog']
    for la in labels: os.makedirs(args.save_path+la, exist_ok=True)
    
    df = pd.read_excel(home_path+'yoga-pose-classification/movement_frames_info.xlsx', sheet_name='train')
    
    v_num = 0
    for video_file in video_files:
        cap = cv2.VideoCapture(os.path.join(data_path, video_file))
        if not cap.isOpened():
            print(f"Error: Could not open video {video_file}")
            continue
        
        video_name = os.path.splitext(video_file)[0]
        rows = df.loc[df['VIDEO_name'] == video_name]
        rows.reset_index(drop=True)
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(os.path.join(args.save_path, video_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
        
        # Process each frame of the video
        count = 0
        index = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Load the pre-trained model
            model = fasterrcnn_resnet50_fpn(pretrained=True)
            model.eval()  # Set the model to evaluation mode
            
            # Perform object detection on the frame
            predictions = detect_objects(model, frame)
            frame_with_boxes, cropped_frame = visualize_result(frame, predictions)
            out.write(frame_with_boxes)
            
            # svae the cropped frame
            if cropped_frame!=None and count >= rows.iloc[index, 2] and count <= rows.iloc[index, 3]:
                pose = rows.iloc[index, 1]
                class_path = video2class(pose, args.save_path, v_num)
                if count==rows.iloc[index, 3] : 
                    index+=1
                    v_num+=1
                    print(f'fisnied video {video_name} pose={pose}')
                if index==12: 
                    print(f'----------finished video {video_name}----------')
                    break
                cv2.imwrite(f'{class_path}{video_name}_{count}.png', cropped_frame)

            # Display the frame (optional)
            cv2.imshow('Object Detection', frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            count+=1

        # Release video capture and writer objects
        cap.release()
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
