import pandas as pd             
import matplotlib.pyplot as plt   

home = '/Users/chenyuru/Desktop/Yoga/output'
dist = '/camera_test/'+'5'
path = home+dist
front = pd.read_csv( path +'/front.csv' )
lfront = pd.read_csv( path +'/lfront.csv' )
rfront = pd.read_csv( path +'/rfront.csv' )
# left = pd.read_csv( path +'/left.csv' )
# right = pd.read_csv( path +'/right.csv' )

title = ['Ankle', 'Knee', 'Hip', 'Shoulder', 'Elbow', 'Wrist']

points = ['l_ankle', 'r_ankle', 'l_knee', 'r_knee', 'l_hip', 'r_hip', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist', 'r_wrist']
fps = 29
# front
for i in range(0, len(points), 2):
    plt.figure(figsize=(30,15), dpi=100, linewidth = 8)
    # plt.plot(front["time"], front[points[i]], linestyle='-', color='grey', label='front '+points[i])
    
    # SMA
    front_l_sma = front[points[i]].rolling(window=fps).mean()
    plt.plot(front["time"], front_l_sma, label='left')
    
    front_r_sma = front[points[i+1]].rolling(window=fps).mean()
    plt.plot(front["time"], front_r_sma, label='right')

    plt.xlabel('time', size=20)
    plt.ylabel('degree', size=20)
    plt.title(title[i//2]+' from front', size=30)
    plt.legend(loc = 'upper right', fontsize=20)
    plt.savefig(path + '/front_' + title[i//2] + '.png')
    plt.close()


# l front
for i in range(0, len(points), 2):
    plt.figure(figsize=(30,15), dpi=100, linewidth = 8)
    # plt.plot( lfront["time"], lfront[ points[i] ], linestyle='-', color='grey', label='l_front '+points[i] )
    
    lfront_f_sma = lfront[points[i]].rolling(window=fps).mean()
    plt.plot( lfront["time"], lfront_f_sma, label='left')
    
    lfront_r_sma = lfront[points[i+1]].rolling(window=fps).mean()
    plt.plot( lfront["time"], lfront_r_sma, label='right')
    
    plt.xlabel('time', size=20)
    plt.ylabel('degree', size=20)
    plt.title(title[i//2]+' from left front', size=30)
    plt.legend(loc = 'upper right', fontsize = 20)
    plt.savefig(path + '/lfront_' + title[i//2] + '.png')
    plt.close()

# r front
for i in range(0, len(points), 2):
    plt.figure(figsize=(30,15), dpi=100, linewidth = 8)
    # plt.plot( rfront["time"], rfront[ points[i] ], linestyle='-', color='grey', label='r_front '+points[i] )
    
    rfront_l_sma = rfront[points[i]].rolling(window=fps).mean()
    plt.plot( rfront["time"], rfront_l_sma, label='left')
    
    rfront_r_sma = rfront[points[i+1]].rolling(window=fps).mean()
    plt.plot( rfront["time"], rfront_r_sma, label='right')
    

    plt.xlabel('time', size=20)
    plt.ylabel('degree', size=20)
    plt.title(title[i//2]+' from right front', size=30)
    plt.legend(loc = 'upper right', fontsize = 20)
    plt.savefig(path + '/rfront_' + title[i//2] + '.png')
    plt.close()


# left
# for i in range(0, len(points),2):
#     plt.figure(figsize=(30,15), dpi=100, linewidth = 8)
#     # plt.plot( left["time"], left[ points[i] ], linestyle='-', color='grey', label='left '+points[i])
    
#     left_l_sma = left[points[i]].rolling(window=fps).mean()
#     plt.plot( left["time"], left_l_sma, label='left')
    
#     left_r_sma = left[points[i+1]].rolling(window=fps).mean()
#     plt.plot( left["time"], left_r_sma, label='right')
    

#     plt.xlabel('time', size=20)
#     plt.ylabel('degree', size=20)
#     plt.title(title[i//2]+' from left', size=30)
#     plt.legend(loc = 'upper right', fontsize = 20)
#     plt.savefig(path+ '/left_' + title[i//2] + '.png')
#     plt.close()


# right
# for i in range(0, len(points), 2):
#     plt.figure(figsize=(30,15), dpi=100, linewidth = 8)
#     # plt.plot( right["time"], right[ points[i] ], linestyle='-', color='grey', label='right '+points[i])
    
#     right_l_sma = right[points[i]].rolling(window=fps).mean()
#     plt.plot( right["time"], right_l_sma, label='left')

#     right_r_sma = right[points[i+1]].rolling(window=fps).mean()
#     plt.plot( right["time"], right_r_sma, label='right')
    

#     plt.xlabel('time', size=20)
#     plt.ylabel('degree', size=20)
#     plt.title(title[i//2]+' from right', size=30)
#     plt.legend(loc = 'upper right', fontsize = 20)
#     plt.savefig(home + '/output/all_angles/right/smoothing_' + title[i//2] + '.png')
#     plt.close()

# all
# for i in range(len(points)):
#     plt.figure(figsize=(30,15), dpi=100)
   
#     front_sma = front[points[i]].rolling(window=fps).mean()
#     # plt.plot(front["time"], front_sma, label='front')
    
#     lfront_sma = lfront[points[i]].rolling(window=fps).mean()
#     # plt.plot( lfront["time"], lfront_sma, label='left_front')

#     rfront_sma = rfront[points[i]].rolling(window=fps).mean()
#     plt.plot( rfront["time"], rfront_sma, label='right_front')

#     left_sma = left[points[i]].rolling(window=fps).mean()
#     # plt.plot( left["time"], left_sma, label='left')

#     right_sma = right[points[i]].rolling(window=fps).mean()
#     plt.plot( right["time"], right_sma, label='right')
    
#     plt.xlabel('time', size=20)
#     plt.ylabel('degree', size=20)
#     plt.title(points[i], size=30)
#     plt.legend(loc = 'upper right', fontsize=20)
#     plt.savefig(home + '/output/all_angles/r+rfr/' + points[i] + '.png')
#     plt.close()

