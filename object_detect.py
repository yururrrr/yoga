import cv2
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import argparse
import os
import pandas as pd
import time
# from utils._tool import crop

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='video folder path')
    parser.add_argument('--save_path', type=str, help='save folder path')

    return parser.parse_args()

# Preprocess input image
def preprocess_image(image):
    image_tensor = F.to_tensor(image)
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
    return image_tensor

# Perform inference
def detect_objects(model, frame):
    # Convert the frame to a PyTorch tensor
    image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(device)  # Convert to CHW format
    image_tensor /= 255.0  # Normalize to [0, 1]
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions

# Visualize results
def visualize_result(frame, predictions):
    max_area = 0
    min_x = 2000
    selected_box = 0
    max_width = 640
    
    for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
        xmin, ymin, xmax, ymax = box.tolist()  # Convert tensor to list
        xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)  # Convert to integers
        cen_x = xmax-xmin
        if label.item() == 1 and score.item() > 0.5:  # Filter for 'person' class
            area = (xmax - xmin)*(ymax - ymin)
            if area > max_area and xmin < min_x:
                max_area = area
                min_x = xmin
                selected_box = (xmin, ymin, xmax, ymax)

    square_x1, square_y1, square_x2, square_y2 = 0, 0, 0, 0
    if selected_box:
        xmin, ymin, xmax, ymax = selected_box            
        # Calculate the larger square bounding box
        width = xmax - xmin
        height = ymax - ymin
        max_dim = max(width, height)
        center_x = xmin + width // 2
        center_y = ymin + height // 2
        square_x1 = center_x - max_dim // 2 - max_dim // 4
        square_y1 = center_y - max_dim // 2 - max_dim // 4
        square_x2 = square_x1 + max_dim + max_dim // 2
        square_y2 = square_y1 + max_dim + max_dim // 2

        frame = cv2.rectangle(frame, (square_x1, square_y1), (square_x2, square_y2), (0, 255, 0), 2)
        
        return frame, [square_x1, square_y1, square_x2, square_y2]
    else : return frame, None

def video2class(pose, v_num, save_path):
    
    if pose ==1:    output_frame_path = f'mountain pose/{v_num}/'
    elif pose ==2:  output_frame_path = f'raised hands pose/{v_num}/'
    elif pose ==3:  output_frame_path = f'standing forward bend/{v_num}/'
    elif pose==4:   output_frame_path = f'half standing forward bend/{v_num}/'
    elif pose==5:   output_frame_path = f'plank/{v_num}/'
    elif pose==6:   output_frame_path = f'four-limbed staff pose/{v_num}/'
    elif pose==7:   output_frame_path = f'upward facing dog/{v_num}/'
    elif pose==8:   output_frame_path = f'downward facing dog/{v_num}/'
    
    out = os.path.join(save_path, output_frame_path)
    os.makedirs(out, exist_ok=True)
    return out

def main():
    args = parse_args()
    
    
    home_path = 'C:/Users/VIP/Desktop/yuru/yoga/'
    data_path = os.path.join(home_path, args.data_path)
    save_path = os.path.join(home_path, args.save_path)
    
    if args.data_path is None or not os.path.exists(data_path):
        print("Please provide a valid data path.")
        print(data_path)
        return
    os.makedirs((save_path), exist_ok=True)

    video_files = [f for f in os.listdir(data_path) if f.endswith('.MP4')]

    labels=['downward facing dog', 'four-limbed staff pose', 'half standing forward bend', 'mountain pose', 'plank', 'raised hands pose', 'standing forward bend', 'upward facing dog']
    for la in labels: os.makedirs(os.path.join(home_path, args.save_path, la), exist_ok=True)
    
    df = pd.read_excel(home_path+'yoga-pose-classification/movement_frames_info.xlsx', sheet_name='train')
    
    v_num = 0
    start_time = time.time()
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval()  # Set the model to evaluation mode
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
        out = cv2.VideoWriter(os.path.join(save_path, video_file), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
   
        # Process each frame of the video
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # if frame_count % 15 != 0:
            #     continue
            # Perform object detection on the frame
            if frame_count % 15 == 0:
                predictions = detect_objects(model, frame)
            frame_with_boxes, box = visualize_result(frame, predictions)
            
            # svae the cropped frame
            # if box is not None:  # Check if frame_with_boxes is not None
            #     if count >= rows.iloc[index, 2] and count <= rows.iloc[index, 3]:
            #         pose = rows.iloc[index, 1]
            #         class_path = video2class(pose, v_num, save_path)
            #         if count == rows.iloc[index, 3]:
            #             index += 1
            #             v_num += 1
            #             print(f'Finished video {video_name}, pose={pose}')
            #         if index == 12: 
            #             break

            #         cropped_frame = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            #         # print(box)
            #         # print(cropped_frame)
            #         if cropped_frame is not None:  # Check if cropped_frame is not None
            #             # print('here')
            #             if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:
            #                 path = os.path.join(class_path, f'{video_name}_{count}.png')
            #                 # print(path)
            #                 cv2.imwrite(path, cropped_frame)
            
            
            out.write(frame_with_boxes)
            cv2.imshow('Object Detection', frame_with_boxes)

            # Display the frame (optional)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            frame_count+=1

        # Release video capture and writer objects
        cap.release()
        out.release()
        end_time = time.time()
        print(f'----------finished video {video_name}----------{end_time-start_time:.2f} seconds')
        print()
        frame_count+=1
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
