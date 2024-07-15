import cv2
import numpy as np
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, random_split
from models import CNN_LSTM_frame, CNN_LSTM_clip
from dataset import VideoDataset
from utils._tool import  split_dataset, visualize_test_results
# import time
import argparse
import os
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torch.nn.functional as F

# import mediapipe as mp
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.manual_seed(42)
labels = ['1.mountain pose', '2.raised hands pose', '3.standing forward bend', '4.half standing forward bend', '5.plank', '6.four-limbed staff pose', '7.upward facing dog', '8.downward facing dog']
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='VIDEO', type=str, help='frames folder path')
    parser.add_argument('--save_path', default='output/test_results/OD', type=str, help='save folder path')
    parser.add_argument('--height', default=112, type=int, help='height of the input image')
    parser.add_argument('--width', default=112, type=int, help='width of the input image')
    parser.add_argument('--batch_size', default=8, type=int, help='batch size for training')
    parser.add_argument('--cnn_net', default='resnet18', type=str, help='resnet18, resnet34, resnet50')
    parser.add_argument('--num_classes', default=8, type=int, help='number of classes')
    parser.add_argument('--cnn_hidden_size', default=2048, type=int, help='hidden size of the CNN')
    parser.add_argument('--lstm_hidden_size', default=256, type=int, help='hidden size of the LSTM')
    parser.add_argument('--model', default='model/resnet18_LSTM_batch32_clip.pt', type=str, help='path to the trained model')

    return parser.parse_args()

def preprocess(height, width, data_path):
    
    data_transform = transforms.Compose([
        transforms.Resize((height, width)),     #Resize
        transforms.ToTensor(),                  #Convert to Pytorch tensors
    ])

    dataset = VideoDataset(root_dir=data_path, transform=data_transform)
    return dataset
    # class_to_indicies = to_indices(dataset)

def detect_objects(model, frame):
    # Convert the frame to a PyTorch tensor
    image_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(device)  # Convert to CHW format
    image_tensor /= 255.0  # Normalize to [0, 1]
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    return predictions

global max_area, selected_box, prevx, prevy

def visualize_result(frame, predictions, model_name):

    global max_area, selected_box, prevx, prevy
    frame_height, frame_width = frame.shape[:2]
    
    if model_name == 'fasterrcnn_resnet50_fpn':
        for box, label, score in zip(predictions[0]['boxes'], predictions[0]['labels'], predictions[0]['scores']):
            xmin, ymin, xmax, ymax = box.tolist()  # Convert tensor to list
            xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)  # Convert to integers
            prevx = (xmin + xmax) // 2
            prevy = (ymin + ymax) // 2

            if label.item() == 1 and score.item() > 0.5:  # Filter for 'person' class
                area = (xmax - xmin)*(ymax - ymin)
                if area > max_area:
                    max_area = area
                    selected_box = (xmin, ymin, xmax, ymax)

    elif model_name =='mediapipe':
        for box in predictions:
            xmin, ymin, xmax, ymax = box
            # xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)  # Convert to integers
            # # prevx = (xmin + xmax) // 2
            # # prevy = (ymin + ymax) // 2
            
            area = (xmax - xmin)*(ymax - ymin)
            if area > max_area:
                max_area = area
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

        # if abs(prevx-center_x) >50 :center_x = prevx
        # if abs(prevy-center_y) >50 :center_y = prevy
        
        square_x1 = max(0, min(center_x - 3 * max_dim // 4, frame_width))
        square_y1 = max(0, min(center_y - 3 * max_dim // 4, frame_height))
        square_x2 = max(0, min(center_x + 3 * max_dim // 4, frame_width))
        square_y2 = max(0, min(center_y + 3 * max_dim // 4, frame_height))

        frame = cv2.rectangle(frame, (square_x1, square_y1), (square_x2, square_y2), (0, 255, 0), 2)
        
        return frame, [square_x1, square_y1, square_x2, square_y2]
    else : return frame, None



def test_frames(data_path, save_path, args):

    dataset = preprocess(args.height, args.width, data_path)

    split_ratio = 0.8  #80% for training, 20% for valid
    train_dataset, valid_dataset, test_dataset = split_dataset(dataset, split_ratio)

    model = CNN_LSTM_frame(num_classes=8, cnn_hidden_size=512, lstm_hidden_size=256, net='resnet50')
    model.load_state_dict(torch.load(args.model))
    # model.eval()
    
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Test the model and save results
    visualize_test_results(model, test_dataloader, save_path)

def predict_and_save_video(data_path, model, preprocess, output_path):
    
    cap = cv2.VideoCapture(data_path)
    
    # Get the width, height, and FPS of the original video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    predictions = []
    od_model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    od_model.eval()  # Set the model to evaluation mode
    print('Load model: fasterrcnn_resnet50_fpn')
    
    
    # print('ready')
    global max_area, selected_box, prevx, prevy
    max_area, prevx, prevy = 0, 0, 0
    selected_box = None 
    while cap.isOpened():
        ret, frame = cap.read()
        # print('readed')
        if not ret:
            break
        # print('in')
        od_results = detect_objects(od_model, frame)
        # print('OD')
        frame_with_boxes, box = visualize_result(frame, od_results, 'fasterrcnn_resnet50_fpn')
        # print('OD')
        if box is not None:
            cropped_frame = frame[int(box[1]):int(box[3]), int(box[0]):int(box[2])]
            
            if cropped_frame.shape[0] > 0 and cropped_frame.shape[1] > 0:

                # Preprocess the frame
                input_tensor = preprocess(cropped_frame)
                # print(input_tensor.shape)
                input_batch = input_tensor.unsqueeze(0).unsqueeze(0)  # Create a mini-batch as expected by the model  
                # print(input_batch.shape)
                if torch.cuda.is_available():
                    input_batch = input_batch.to(device)
                    model.to(device)
                # Perform inference
                with torch.no_grad():
                    output = model(input_batch)
                    probabilities = F.softmax(output, dim=1)
        
                    # Get the predicted class (assuming single-label classification)
                    max_prob, predicted = torch.max(probabilities, 1)
                    predicted_class = predicted.item()
                    predictions.append(predicted_class)

                    # Add the prediction text to the frame
                    if max_prob.item() > 0.9:
                        cv2.putText(frame_with_boxes, f"Predicted: {labels[predicted_class]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)

                    # Write the frame with prediction to the output video
                    out.write(frame_with_boxes)
        frame_idx += 1

    # Release the video capture and writer objects
    cap.release()
    out.release()

    print(f"Video saved to {output_path}")


def test_videos(data_path, save_path, args):
    
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((args.height, args.width)),     #Resize
        transforms.ToTensor(),                  #Convert to Pytorch tensors
    ])
    # video_data = VideoReader(data_path, 
    #                          transform=transforms.Compose([
    #                             transforms.Resize((args.height, args.width)),     #Resize
    #                             transforms.ToTensor(),                  #Convert to Pytorch tensors
    #                         ]))
    # dataset = preprocess(video_data, args.height, args.width)
    # print(video_data.__len__())
    model = CNN_LSTM_clip(num_classes=8, cnn_hidden_size=args.cnn_hidden_size, lstm_hidden_size=256, net='resnet18')
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model))
    model.to(device)
    model.eval()

    # List to store predictions
    predict_and_save_video(data_path, model, preprocess, save_path)

if __name__ == '__main__':

    home_path = ''

    args = parse_args()
    data_path = os.path.join(home_path, args.data_path)
    save_path =  os.path.join(home_path, args.save_path)#, args.data_path.split('/')[-1].split('.')[0])
    # print(f'detect: {data_path}')
    
    os.makedirs(save_path, exist_ok=True)
    video_files = [f for f in os.listdir(data_path) if f.endswith('.MP4') or f.endswith('.mp4')]
    print(video_files)
    for video_file in video_files:
        video_path = os.path.join(data_path, video_file)
        save_video_path = os.path.join(save_path, video_file.split('.')[0]+'_test.mp4')
        print(f'video_path: {video_path}')

        # test_frames(data_path, save_path, args)
        test_videos(video_path, save_video_path, args)
        # print(f'save in {save_video_path}')





# test rvm
# def test_rvm():
    
#     video_name = ['1_left_front']
#     home_path = 'C:/Users/VIPLAB/Desktop/yuru/'
#     video_path = home_path+'RVM_output/1_left_front.mp4'

#     cap = cv2.VideoCapture(video_path)

#     frame_count = 0
#     start_time = None
#     pose_sequence = []
#     min_time_threshold = 10
#     min_frame_threshlod = 30

#     def all_same(sequence):
#         return all(x == sequence[0] for x in sequence)

#     width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#     height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#     fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#     out = cv2.VideoWriter(f'{home_path}/output/{video_name[0]}_test.mp4', fourcc, 30, (width, height))

#     start = time.time()

#     while(cap.isOpened()):
#         ret, frame = cap.read()
#         if not ret:
#             print('Ignore empty camera frame.')
#             break
#         tensor_frame = transforms.ToTensor()(frame).unsqueeze(0)
#         with torch.no_grad():
#             output = model(tensor_frame)
#             _, predict = output.max(1)
#             predicted_pose = predict.item()
            
#         # frame_count += 1
#         # pose_sequence.append(predicted_pose)
        
#         # if start_time is None:
#         #     start_time = cap.get(cv2.CAP_PROP_POS_MSEC)
#         # else:
#         #     current_time = cap.get(cv2.CAP_PROP_POS_MSEC)
#         #     time_difference = current_time - start_time 
#         #     if time_difference >= min_time_threshold:
#         #         if len(pose_sequence) >= min_frame_threshlod:
#         #             if all_same(pose_sequence):
#         cv2.putText(frame, str(labels[predicted_pose]), (50, 50) ,
#             cv2.FONT_HERSHEY_SIMPLEX , 1, (255, 0, 0), 2)
#         out.write(frame)
        
#     out.release()

#     end = time.time()
#     execution = end-start
#     print('Execution time:',execution, 's')
    