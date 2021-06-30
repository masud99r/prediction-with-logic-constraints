import numpy as np


data_dir = "../logic_constraints"
if __name__ == "__main__":
	
	# '''
	X = np.loadtxt(data_dir+"/taurus_kinematics_x.txt", delimiter=' ')
	y = np.loadtxt(data_dir+"/taurus_kinematics_y.txt", delimiter=' ')
	y = y - 1 # make the class label starts from 0
	y = y.reshape(-1, 1)
	Xy = np.concatenate((X, y), axis=1)
	targetlen = 1405
	np.savetxt(data_dir+"/taurus_kinematics_train.txt", Xy[:targetlen], delimiter=' ')
	np.savetxt(data_dir+"/taurus_kinematics_test.txt", Xy[targetlen:], delimiter=' ')
	# '''


	# '''
	Xy = np.loadtxt(data_dir+"/gym_pickplace_kinematics.txt", delimiter=' ')
	targetlen = 126
	np.savetxt(data_dir+"/gym_kinematics_train.txt", Xy[:targetlen], delimiter=' ')
	np.savetxt(data_dir+"/gym_kinematics_test.txt", Xy[targetlen:], delimiter=' ')
	
	# '''