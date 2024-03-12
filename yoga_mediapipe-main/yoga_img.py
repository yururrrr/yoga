import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

mp_image = []
detection_result = []
annotated_image = []

# Create an PoseLandmarker object.
base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)


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


def cal_angle(point_a , point_b, point_c):
    # 向量
    x1, y1, z1 = (point_a.x - point_b.x), (point_a.y - point_b.y), (point_a.z - point_b.z)
    x2, y2, z2 = (point_c.x - point_b.x), (point_c.y - point_b.y), (point_c.z - point_b.z)

    cos_b = (x1*x2 + y1*y2 + z1*z2) / (math.sqrt(x1**2 + y1**2 + z1**2) * math.sqrt(x2**2 + y2**2 + z2**2))
    B = math.degrees(math.acos(cos_b))

    return B



# input images


#  Detect pose landmarks from the input image.


points = ['L_ankle', 'R_ankle', 'L_knee', 'R_knee', 'L_hip', 'R_hip', 'L_shoulder', 'R_shoulder', 'L_elbow', 'R_elbow', 'L_wrist', 'R_wrist']
r_ankle = []
l_ankle = []
r_knee = []
l_knee = []
l_hip = []
r_hip = []
l_shoulder = []
r_shoulder = []
r_elbow = []
l_elbow = []
r_wrist = []
l_wrist = []
import PIL.Image

# for i in range(1,8):
#     rgba_image = PIL.Image.open('/Users/chenyuru/Desktop/Yoga/sample/sample2_images/'+str(i)+'.jpg')
#     rgb_image = rgba_image.convert('RGB')
#     rgb_image.save('/Users/chenyuru/Desktop/Yoga/sample/sample2_images/'+str(i)+'.jpg')

for i in range(1,13):
    # image = cv2.imread('/Users/chenyuru/Desktop/Yoga/sample/sample2_images/'+str(i)+'.jpg')
    # print(image.shape)
    mp_image.append(mp.Image.create_from_file('/Users/chenyuru/Desktop/Yoga/movements/step'+str(i)+'.jpeg'))
    # print(mp_image[i-1].numpy_view())
    detection_result.append(detector.detect(mp_image[i-1]))
    annotated_image.append(draw_landmarks_on_image(mp_image[i-1].numpy_view(), detection_result[i-1]))
    
    r_ankle.append( cal_angle(detection_result[i-1].pose_landmarks[0][28], detection_result[i-1].pose_landmarks[0][30], detection_result[i-1].pose_landmarks[0][32]) )
    l_ankle.append( cal_angle(detection_result[i-1].pose_landmarks[0][27], detection_result[i-1].pose_landmarks[0][29], detection_result[i-1].pose_landmarks[0][31]) )
    r_knee.append( cal_angle(detection_result[i-1].pose_landmarks[0][24], detection_result[i-1].pose_landmarks[0][26], detection_result[i-1].pose_landmarks[0][28]) )
    l_knee.append( cal_angle(detection_result[i-1].pose_landmarks[0][23], detection_result[i-1].pose_landmarks[0][25], detection_result[i-1].pose_landmarks[0][27]) )
    r_hip.append( cal_angle(detection_result[i-1].pose_landmarks[0][12], detection_result[i-1].pose_landmarks[0][24], detection_result[i-1].pose_landmarks[0][26]) )
    l_hip.append( cal_angle(detection_result[i-1].pose_landmarks[0][11], detection_result[i-1].pose_landmarks[0][23], detection_result[i-1].pose_landmarks[0][25]) )
    r_shoulder.append( cal_angle(detection_result[i-1].pose_landmarks[0][14], detection_result[i-1].pose_landmarks[0][12], detection_result[i-1].pose_landmarks[0][24]) )
    l_shoulder.append( cal_angle(detection_result[i-1].pose_landmarks[0][11], detection_result[i-1].pose_landmarks[0][23], detection_result[i-1].pose_landmarks[0][25]) )
    r_elbow.append( cal_angle(detection_result[i-1].pose_landmarks[0][16], detection_result[i-1].pose_landmarks[0][14], detection_result[i-1].pose_landmarks[0][12]) )
    l_elbow.append( cal_angle(detection_result[i-1].pose_landmarks[0][15], detection_result[i-1].pose_landmarks[0][13], detection_result[i-1].pose_landmarks[0][11]) )
    r_wrist.append( cal_angle(detection_result[i-1].pose_landmarks[0][20], detection_result[i-1].pose_landmarks[0][16], detection_result[i-1].pose_landmarks[0][14]) )
    l_wrist.append( cal_angle(detection_result[i-1].pose_landmarks[0][19], detection_result[i-1].pose_landmarks[0][15], detection_result[i-1].pose_landmarks[0][13]) )
        
# print(detection_result[0])
# detection_result.pose_landmarks[0][17].y
# annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
# cv2_imshow(cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))


for i in range(1,13):
   
    img = cv2.cvtColor(annotated_image[i-1],cv2.COLOR_RGB2BGR)
    height, width, b = img.shape
    # print(img.shape)
    cv2.putText(img, str(points[0])+':'+str(round(l_ankle[i-1])), (width-600, 300), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(points[1])+':'+str(round(r_ankle[i-1])), (width-300, 300), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(points[2])+':'+str(round(l_knee[i-1])), (width-600, 250), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(points[3])+':'+str(round(r_knee[i-1])), (width-300, 250), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(points[4])+':'+str(round(l_hip[i-1])), (width-600, 200), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(points[5])+':'+str(round(r_hip[i-1])), (width-300, 200), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(points[6])+':'+str(round(l_shoulder[i-1])), (width-600, 150), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(points[7])+':'+str(round(r_shoulder[i-1])), (width-300, 150), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(points[8])+':'+str(round(l_elbow[i-1])), (width-600, 100), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(points[9])+':'+str(round(r_elbow[i-1])), (width-300, 100), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(points[10])+':'+str(round(l_wrist[i-1])), (width-600, 50), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(img, str(points[11])+':'+str(round(r_wrist[i-1])), (width-300, 50), cv2.FONT_HERSHEY_SIMPLEX,1.2, (0, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imwrite('/Users/chenyuru/Desktop/Yoga/output/movements/out'+str(i)+'.jpg',img)