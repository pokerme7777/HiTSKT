# import library
import pandas as pd
import numpy as np
import random
import math


def read_data_into_array(df, padding_node=0, padding_correct_value=2, action_size=64, session_size=16,
							EOS_quesiton=None, EOS_skill=None, EOS_C_value=3, BOS_c_Value=4,
							session_EOS_question=None, session_EOS_skill=None, session_EOS_c_value=4,
							EOS_q_no=None, EOS_stu_no=None, SEOS_q_no=None, SEOS_stu_no=None):
	''' create array for training

		Args:
			df(pd.dataframe)
			padding_node: padding for skill or problem
			padding_correct_value: padding value for correctness
			action_size: how many actions for each session
			session_size: how many sessions for each train

		Returns:
			all_train_list: the training array
	'''
	# Create training array
	all_train_list = []
	# Create val array
	all_val_list = []
	# Create Test array
	all_test_list = []

	problem_padding_list = [padding_node] * action_size
	correct_padding_list = [padding_correct_value] * action_size

	padding_session = [problem_padding_list, problem_padding_list, correct_padding_list, problem_padding_list,
						problem_padding_list, correct_padding_list]

	EOS_session_problem_list = [session_EOS_question]*(action_size-1)
	EOS_session_problem_list.append(EOS_quesiton)
	EOS_session_skill_list = [session_EOS_skill]*(action_size-1)
	EOS_session_skill_list.append(EOS_skill)
	EOS_session_correct_list = [session_EOS_c_value]*(action_size-1)
	EOS_session_correct_list.append(EOS_C_value)

	EOS_session_qno_list = [SEOS_q_no]*(action_size-1)
	EOS_session_qno_list.append(EOS_q_no)
	EOS_session_stuno_list = [SEOS_stu_no]*(action_size-1)
	EOS_session_stuno_list.append(EOS_stu_no)

	EOS_session = [EOS_session_problem_list, EOS_session_skill_list, EOS_session_correct_list,
					EOS_session_qno_list, EOS_session_stuno_list, EOS_session_correct_list]

	for i in df.studentId.unique():

		# dataframe for one student
		df_student = df[df.studentId == i]

		all_session_array = []

		for j in df_student.session_no.unique():

			# dataframe for one session for this student
			df_session = df_student[df_student.session_no == j]

			# Create empty array for each session
			session_array = []

			# Question array
			question_array = df_session.problemId.tolist()

			# correctness_array
			correct_array = df_session.correct.tolist()

			# correctness_array
			skill_array = df_session.skill.tolist()

			# qno_array
			qno_array = df_session.question_no.tolist()

			# stu_no_array
			stu_no_array = df_session.studentId.tolist()

			if len(question_array) >= action_size:
				question_array = question_array[(-1) * (action_size-1):]
				skill_array = skill_array[(-1) * (action_size-1):]
				correct_array = correct_array[(-1) * (action_size-1):]
				qno_array = qno_array[(-1) * (action_size-1):]
				stu_no_array = stu_no_array[(-1) * (action_size-1):]

			else:
				question_array.extend([padding_node] * (action_size - len(question_array) -1))
				skill_array.extend([padding_node] * (action_size - len(skill_array)-1))
				correct_array.extend([padding_correct_value] * (action_size - len(correct_array)-1))
				qno_array.extend([padding_node] * (action_size - len(qno_array) -1))
				stu_no_array.extend([padding_node] * (action_size - len(stu_no_array) -1))

			question_array.append(EOS_quesiton)
			skill_array.append(EOS_skill)
			correct_array.append(EOS_C_value)
			qno_array.append(EOS_q_no)
			stu_no_array.append(EOS_stu_no)

			enc_correct = correct_array[:-1]
			enc_correct.insert(0, BOS_c_Value)

			# Append value into array for this session
			session_array.append(question_array)
			session_array.append(skill_array)
			session_array.append(correct_array)
			session_array.append(qno_array)
			session_array.append(stu_no_array)
			session_array.append(enc_correct)

			# Append value into session array for this session
			all_session_array.append(session_array)
		
		all_session_array_len = len(all_session_array)
		test_session_len = all_session_array_len // 5
		train_max_session = all_session_array_len - 2*test_session_len

		# Train array 
		for ses_no in range(train_max_session - 1):

			# Create encoder input list
			one_train_list = []

			# get (num_session-1) sessions
			one_train_list.extend(all_session_array[:ses_no + 1])

			# ensure the number of session is equal with session_size
			if len(one_train_list) < session_size:

				one_train_list.extend([padding_session] * (session_size - len(one_train_list) - 1 ))

			else:
				one_train_list = one_train_list[(-1) * (session_size-1):]

			one_train_list.append(EOS_session)

			# append target session
			one_train_list.append(all_session_array[ses_no + 1])

			all_train_list.append(one_train_list)
		
		# val array 		
		for ses_no in range(train_max_session-1, all_session_array_len-test_session_len-1):

			# Create encoder input list
			one_val_list = []

			# get (num_session-1) sessions
			one_val_list.extend(all_session_array[:ses_no + 1])

			# ensure the number of session is equal with session_size
			if len(one_val_list) < session_size:

				one_val_list.extend([padding_session] * (session_size - len(one_val_list) - 1 ))

			else:
				one_val_list = one_val_list[(-1) * (session_size-1):]

			one_val_list.append(EOS_session)

			# append target session
			one_val_list.append(all_session_array[ses_no + 1])

			all_val_list.append(one_val_list)

		# test array 		
		for ses_no in range(all_session_array_len-test_session_len-1, all_session_array_len-1):

			# Create encoder input list
			one_test_list = []

			# get (num_session-1) sessions
			one_test_list.extend(all_session_array[:ses_no + 1])

			# ensure the number of session is equal with session_size
			if len(one_test_list) < session_size:

				one_test_list.extend([padding_session] * (session_size - len(one_test_list) - 1 ))

			else:
				one_test_list = one_test_list[(-1) * (session_size-1):]

			one_test_list.append(EOS_session)

			# append target session
			one_test_list.append(all_session_array[ses_no + 1])

			all_test_list.append(one_test_list)

	return all_train_list, all_val_list, all_test_list