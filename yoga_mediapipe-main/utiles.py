import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions
import os

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

def check_folder(folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

def read_data(data_path):
    all_videos = []
    all_labels = []
    all_keypoints = []
    for class_label, class_name in enumerate(os.listdir(data_path)):
        class_path = os.path.join(PATH, class_name)
        if os.path.isdir(class_path):
            print(class_path)
            for dir in os.listdir(class_path):
                dir_path = os.path.join(class_path, dir)

                # open cvs file
                dirname = dir.split('-')
                csv_path = 'C:/Users/VIP/Desktop/yuru/VIDEO/output/'+ dirname[0] + '/landmarks.csv'
                # print(csv_path)
                pd_keypoints = pd.read_csv(csv_path)
                # print(keypoints.head())
                # break
                images = []
                labels = []
                start_index = -2
                
                for filename in os.listdir(dir_path):
                    if filename.endswith(('.jpg', '.jpeg', '.png')):
                        # read frames
                        # print(dir_path+'/'+filename)
                        img = cv2.imread(dir_path+'/'+filename)
                        img = cv2.resize(img, IMAGE_SIZE)
                        img = img[:, :, [2, 1, 0]]
                        images.append(img)
                        # labels.append(class_name)
                        ori = filename.split('_')
                        index = ori[len(ori)-1].split('.')
                        
                        if start_index == -2: start_index = int(index[0])-1

                # append keypoints
                keypoints = pd_keypoints.iloc[start_index:start_index+len(labels)]

                all_videos.append(images)
                all_labels.append(class_name)
                # key = keypoints.values.tolist
                # print(keypoints.head())
                all_keypoints.append(keypoints.values.tolist())
                # print('len_img = ',len(all_videos), 'shape of this', len(images), 'len_labels', len(all_labels))
                break
                # print(labels[-5:]),

    # print((all_keypoints[0][:5]))
    # print([len(i) for i in all_videos], [len(i) for i in all_keypoints], [len(i) for i in all_labels])
    # return np.array(all_videos, dtype='object'), np.array(all_labels,  dtype='object'), np.array(all_keypoints)
    print('max_length=',max(len(item) for item in all_videos))
    return all_videos, all_labels , all_keypoints

def padded_data(data):
    max_length = max(len(item) for item in data)
    padded = [np.pad(item, ((0, max_length - len(item)), (0, 0), (0, 0), (0, 0)), 'constant') for item in data]
    return padded

def map_back_to_multilabel_structure(one_hot_labels, original_labels):
    mapped_labels = []
    index = 0
    for sublist in original_labels:
        mapped_labels.append(one_hot_labels[index : index + len(sublist)])
        index += len(sublist)
    return mapped_labels


def label_encoder(le, labels, type):
    flattened_labels = [label for sublist in labels for label in sublist]
    if type == 'train':   
        encoded_labels = le.fit_transform(flattened_labels)
    elif type == 'valid':
        encoded_labels = le.transform(flattened_labels)

    return map_back_to_multilabel_structure(to_categorical(encoded_labels), labels)
