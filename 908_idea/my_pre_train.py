# import library
import pandas as pd
import numpy as np


def read_data_into_array(df, padding_node=0, padding_correct_value=2, action_size=64, session_size=16):
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

	problem_padding_list = [padding_node] * action_size
	correct_padding_list = [padding_correct_value] * action_size

	padding_session = [problem_padding_list, problem_padding_list, correct_padding_list, problem_padding_list,
					   problem_padding_list, correct_padding_list]

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

			if len(question_array) >= action_size:
				question_array = question_array[(-1) * action_size:]
				skill_array = skill_array[(-1) * action_size:]
				correct_array = correct_array[(-1) * action_size:]

			else:
				question_array.extend([padding_node] * (action_size - len(question_array)))
				skill_array.extend([padding_node] * (action_size - len(skill_array)))
				correct_array.extend([padding_correct_value] * (action_size - len(correct_array)))

			enc_question = question_array[:-1]
			enc_question.insert(0, padding_node)

			enc_skill = skill_array[:-1]
			enc_skill.insert(0, padding_node)

			enc_correct = correct_array[:-1]
			enc_correct.insert(0, padding_correct_value)

			# Append value into array for this session
			session_array.append(question_array)
			session_array.append(skill_array)
			session_array.append(correct_array)
			session_array.append(enc_question)
			session_array.append(enc_skill)
			session_array.append(enc_correct)

			# Append value into session array for this session
			all_session_array.append(session_array)

		for ses_no in range(len(all_session_array) - 1):

			# Create encoder input list
			one_train_list = []

			# get (num_session-1) sessions
			one_train_list.extend(all_session_array[:ses_no + 1])

			# ensure the number of session is equal with session_size
			if len(one_train_list) < session_size:

				one_train_list.extend([padding_session] * (session_size - len(one_train_list)))

			else:
				one_train_list = one_train_list[(-1) * session_size:]

			# append target session
			one_train_list.append(all_session_array[ses_no + 1])

			all_train_list.append(one_train_list)

	return all_train_list