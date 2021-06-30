import numpy as np
import math

def distance(x1, y1, x2, y2):
	return round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2), 2)
def distance3d(x1, y1, z1, x2, y2, z2):
	return round(math.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2), 2)

def feature_activity(features, labels, robot, fname):
	motion_features = []
	backindex = 1
	for i in range(len(features)):
		motion_feature = []
		# print(features[0])
		# print(len(features[0]))
		lx, ly, lz = features[i][0], features[i][1], features[i][2]
		lgs = features[i][6]
		rx, ry, rz = features[i][7], features[i][8], features[i][9]
		rgs = features[i][13]
		if robot == "Taurus" or robot == "Taurus_sim": # scale between 0, 1. Yumi already in between 0, 1
			lgs =  lgs / 100
			rgs =  rgs / 100
		if i > backindex:
			past_lgs = features[i-backindex][6]
			past_rgs = features[i-backindex][13]
			past_lx, past_ly, past_lz = features[i-backindex][0], features[i-backindex][1], features[i-backindex][2]
			past_rx, past_ry, past_rz = features[i-backindex][7], features[i-backindex][8], features[i-backindex][9]

		else: # make current is the past
			past_lgs = features[i][6]
			past_rgs = features[i][13]
			past_lx, past_ly, past_lz = features[i][0], features[i][1], features[i][2]
			past_rx, past_ry, past_rz = features[i][7], features[i][8], features[i][9]
		
		# get gripper status
		if lgs > 0.90:
			left_gs = 1 #"open"
		elif lgs < 0.35: # 30 is the closing in documentation, +5 for jurking
			left_gs = 2 #"closed"
		elif lgs - past_lgs > 0.05:
			left_gs = 3 #"openning"
		elif past_lgs - lgs > 0.05:
			left_gs = 4 #"closing"
		else:
			left_gs = 0 #"static"

		if rgs > 0.90:
			right_gs = 1 #"open"
		elif rgs < 0.35: # 30 is the closing in documentation, +5 for jurking
			right_gs = 2 #"closed"
		elif rgs - past_rgs > 0.05:
			right_gs = 3 #"openning"
		elif past_rgs - rgs > 0.05:
			right_gs = 4 #"closing"
		else:
			right_gs = 0 #"static"

		# get arm status
		current_arms_distance = distance3d(lx, ly, lz, rx, ry, rz)
		past_arms_distance = distance3d(past_lx, past_ly, past_lz, past_rx, past_ry, past_rz)
		if current_arms_distance - past_arms_distance > 0.001: # check this 5, heavily depend on the motion space
			arm_dist = 1 # increasing
		elif past_arms_distance - current_arms_distance > 0.001:
			arm_dist = 2 # decreasing
		else:
			arm_dist = 0 # static

		motion_feature.append(current_arms_distance)
		motion_feature.append(arm_dist)
		motion_feature.append(lgs)
		motion_feature.append(left_gs)
		motion_feature.append(rgs)
		motion_feature.append(right_gs)
		motion_feature.append(labels[i])
		
		motion_features.append(motion_feature)
		
	print(motion_features[0])
	print(len(motion_features))
	print(len(motion_features[0]))
	np.savetxt(data_dir+'/'+fname+'_motion_features.txt', motion_features, delimiter=' ')
import numpy as np
import math

def feature_activity_gym(features, fname):
	motion_features = []
	backindex = 1
	for i in range(len(features)):
		motion_feature = []
		
		threshold_gs = 0.001
		if i > backindex:
			delta_gs = features[i-1][9] - features[i][9]
			if delta_gs > threshold_gs:
				gs_var =  1 # increasing
			elif delta_gs  < (threshold_gs * -1.0):
				gs_var = 2 # decreasing
			else:
				gs_var = 0 # static
		else:
			gs_var = 0 # static, default
		# keep only true one
		motion_feature.append(gs_var)
		motion_feature.append(features[i][-1])
		motion_features.append(motion_feature)
		
	print(motion_features[0])
	print(len(motion_features))
	print(len(motion_features[0]))
	np.savetxt(data_dir+'/'+fname+'_motion_features.txt', motion_features, delimiter=' ')
		
		
data_dir = "../logic_constraints"
if __name__ == '__main__':
	
	robot = "Taurus"
	Xy = np.loadtxt(data_dir+"/taurus_kinematics_train.txt", delimiter=' ')
	X, y = Xy[:, :-1], Xy[:, -1]
	feature_activity(X, y, robot, fname="taurus_kinematics_train")
	
	Xy = np.loadtxt(data_dir+"/taurus_kinematics_test.txt", delimiter=' ')
	X, y = Xy[:, :-1], Xy[:, -1]
	feature_activity(X, y, robot, fname="taurus_kinematics_test")

	Robot = "Gym"
	Xy = np.loadtxt(data_dir+"/gym_kinematics_train.txt", delimiter=' ')
	feature_activity_gym(Xy, fname="gym_kinematics_train")

	Xy = np.loadtxt(data_dir+"/gym_kinematics_test.txt", delimiter=' ')
	feature_activity_gym(Xy, fname="gym_kinematics_test")

		