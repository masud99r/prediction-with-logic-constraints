import numpy as np

# def gym_convert_to_mln_dbformat(data_dir, datafile):
#     data = np.loadtxt(datafile, delimiter=' ')
#     fw_db = open(data_dir+"/mln_data/"+"gym_pickplace.db", "w")
#     for i in range(len(data)):
#         if i > 500:
#             break
#         single_row_feature = data[i]
#         gs = int(single_row_feature[0])
#         label = "C"+str(int(single_row_feature[-1]))
#         # print(arms_dist, left_gs, right_gs, label)
#         fw_db.write('Has("gs_'+str(gs)+'","'+str(i)+'")\n')
#         # fw_db.write('Has("left_gs_'+str(left_gs)+'","'+str(i)+'")\n')
#         # fw_db.write('Has("right_gs_'+str(right_gs)+'","'+str(i)+'")\n')
#         fw_db.write('Topic("'+str(label)+'","'+str(i)+'")\n')
#     fw_db.close()
def convert_to_mln_dbformat(robot, dfile, savefile):
	data = np.loadtxt(dfile, delimiter=' ')
	fw_db = open(savefile, "w")
	for i in range(len(data)):
		single_row_feature = data[i]
		if robot == "Gym":
			gs = int(single_row_feature[0])
			label = "C"+str(int(single_row_feature[-1]))
			fw_db.write('Has("gs_'+str(gs)+'","'+str(i)+'")\n')
		else:
			arms_dist = int(single_row_feature[1])
			left_gs = int(single_row_feature[3])
			right_gs = int(single_row_feature[5])
			label = "C"+str(int(single_row_feature[-1]))
			fw_db.write('Has("arms_dist_'+str(arms_dist)+'","'+str(i)+'")\n')
			fw_db.write('Has("left_gs_'+str(left_gs)+'","'+str(i)+'")\n')
			fw_db.write('Has("right_gs_'+str(right_gs)+'","'+str(i)+'")\n')
			fw_db.write('Has("left_right_gs_'+str(left_gs)+"_"+str(right_gs)+'","'+str(i)+'")\n')
		fw_db.write('Topic("'+str(label)+'","'+str(i)+'")\n')
	fw_db.close()

data_dir = "../logic_constraints"

if __name__ == '__main__':

	convert_to_mln_dbformat("Taurus", data_dir+"/taurus_kinematics_train_motion_features.txt", data_dir+'/taurus_kinematics_train.db')
	convert_to_mln_dbformat("Taurus", data_dir+"/taurus_kinematics_test_motion_features.txt", data_dir+'/taurus_kinematics_test.db')
	
	convert_to_mln_dbformat("Gym", data_dir+"/gym_kinematics_train_motion_features.txt", data_dir+'/gym_kinematics_train.db')
	convert_to_mln_dbformat("Gym", data_dir+"/gym_kinematics_test_motion_features.txt", data_dir+'/gym_kinematics_test.db')
	

	print("Done")

