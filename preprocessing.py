import pandas as pd
import numpy as np
import math
from dask import dataframe as dd
import argparse

def read_2017(filename='./Dataset/ASSISTment2017_full.csv'):
    ''' Reads the 2017 data into pandas dataframe

    Args:
        filename (str, optional): the path to the file, default is './Dataset/ASSISTment2017_full.csv'

    Returns:
        the dataframe of the ASSISTment 2017 dataset
    '''

    # Dataset is now stored in a Pandas Dataframe (2017)
    df_ori = pd.read_csv(filename, low_memory=False)

    # Selecting the necessary columns
    df_select = (df_ori[
        ['studentId', 'action_num', 'skill', 'problemId', 'problemType', 'assignmentId', 'assistmentId', 'startTime',
         'endTime', 'timeTaken', 'correct']]).copy()


    # Drop the timeover records.
    df_select = df_select[df_select.timeTaken != 9999]

    # Sorting
    df_select = df_select.sort_values(by=['studentId', 'action_num'])

    # Reset index
    df_select = df_select.reset_index(drop=True)

    return df_select


def read_2012(filename='./Dataset/ASSISTment2012_full.csv'):
    ''' Read the 2012 dataset into pandas dataframe

    Args:
        filename (str, optional): the path to the file, default is './Dataset/ASSISTment2012_full.csv'

    Returns:
        (pd.DataFrame): the dataframe of the ASSISTment 2012 dataset
    '''
    df_ori = pd.read_csv(filename, low_memory=False)

    # Select the necessary columns
    df_select = pd.DataFrame()
    df_select['studentId'] = df_ori['user_id']
    df_select['skill'] = df_ori['skill']
    df_select['problemType'] = df_ori['problem_type']
    df_select['problemId'] = df_ori['problem_id']
    df_select['correct'] = df_ori['correct']

    # The original value is a date string in the format of 'YYYY-mm-dd\nHH:MM:SS'
    # We need to convert it into datetime type
    df_select['startTime'] = pd.to_datetime(df_ori['start_time']).astype(int) // (10 ** 9)  # to second
    df_select['endTime'] = pd.to_datetime(df_ori['end_time']).astype(int) // (10 ** 9)  # to second

    # Calculate the time taken for the interaction
    # The substraction of two datetime object is by default in nanosecond
    # So we need to convert it from nanosecond to second
    df_select['timeTaken'] = df_select['endTime'] - df_select['startTime']

    # Drop nan/null rows in the skill column
    df_select = df_select[df_select.skill.notnull()]

    # Initialization of correct value: all the values less than 1 will be 0.
    df_select['correct'] = df_select['correct'].apply(lambda x: 0 if x < 1 else 1)

    # Sort the dataframe using studentid, starttime (Ascending)
    df_select = df_select.sort_values(by=['studentId', 'startTime'])

    # Delete all the rows in which timeTaken is too big
    df_select = df_select[df_select.timeTaken < 9999]

    # Reset index
    df_select = df_select.reset_index(drop=True)

    return df_select

def read_Junyi(filename='./Dataset/Junyi.csv'):
    ''' Read the Junyi dataset into pandas dataframe

    Args:
        filename (str, optional): the path to the file, default is './Dataset/Junyi.csv'

    Returns:
        (pd.DataFrame): the dataframe of the Junyi dataset
    '''
    df_ori = pd.read_csv(filename, low_memory=False)

    # Select the necessary columns
    df_select = pd.DataFrame()
    df_select['studentId'] = df_ori['uuid']
    df_select['skill'] = df_ori['ucid']
    df_select['problemId'] = df_ori['upid']
    df_select['correct'] = df_ori['is_correct']

    # The original value is a date string in the format of 'YYYY-mm-dd\nHH:MM:SS'
    # We need to convert it into datetime type
    df_select['startTime'] = pd.to_datetime(df_ori['timestamp_TW']).astype(int) // (10 ** 9)  # to second
    df_select['endTime'] = df_select['startTime']  # to second

    # Drop nan/null rows in the skill column
    df_select = df_select[df_select.skill.notnull()]

    # Initialization of correct value: all the values less than 1 will be 0.
    df_select['correct'] = df_select['correct'].apply(lambda x: 1 if x else 0)

    # Sort the dataframe using studentid, starttime (Ascending)
    df_select = df_select.sort_values(by=['studentId', 'startTime'])

    # Reset index
    df_select = df_select.reset_index(drop=True)

    return df_select

def read_ednet(filename='./Dataset/ednet.csv'):
    ''' Read the EdNet dataset into dask dataframe

    Args:
        filename (str, optional): the path to the file, default is './Dataset/ednet.csv'

    Returns:
        (pd.DataFrame): the dataframe of the ednet dataset
    '''
    df_ori = dd.read_csv(filename, low_memory=False)

    # Select the necessary columns
    df1 = df_ori[['user_id','content_id','timestamp','task_container_id','answered_correctly']]
    df1 = df1.compute()
    df1 = df1.rename(columns={"user_id": "studentId", "task_container_id": "skill", "content_id": "problemId", "answered_correctly": "correct", "timestamp": "startTime"})

    # The original value is a date string in the format of 'YYYY-mm-dd\nHH:MM:SS'
    # We need to convert it into datetime type
    df1['endTime'] = df1['startTime']  # to second

    # Drop nan/null rows in the skill column
    df1 = df1[df1.skill.notnull()]

    # Initialization of correct value: all the values less than 1 will be 0.
    df1['correct'] = df1['correct'].apply(lambda x: 1 if x else 0)

    # Sort the dataframe using studentid, starttime (Ascending)
    df1 = df1.sort_values(by=['studentId', 'startTime'])

    # Reset index
    df1 = df1.reset_index(drop=True)

    return df1


def factorize_skill_problem(df):
    '''To factorize the skill and problem columns

    Args:
        df (pd.DataFrame)

    Returns:
        df (pd.DataFrame)
    '''
    skill_factorize, skill_unique = pd.factorize(df.skill)
    df.skill = skill_factorize + 1

    problem_factorize, problem_unique = pd.factorize(df.problemId)
    df.problemId = problem_factorize + 1

    student_factorize, student_unique = pd.factorize(df.studentId)
    df.studentId = student_factorize + 1

    return df

def session_division(df, hour=10, minute=0, second=0, ms_state=False):
    ''' Split the learning sequence into different sessions

    Args:
        df (pd.DataFrame):
        hour (int, optional):
        minute (int, optional):
        second (int, optional): the

    Returns:
        (pd.DataFrame): the dataframe with session id
    '''

    df = df.sort_values(by=['studentId', 'startTime'])
    df = df.reset_index(drop=True)
    # Create empty session no list
    session_list = []

    # Set the length of each session
    session_gap = hour * 3600 + minute * 60 + second
    if ms_state:
        session_gap = session_gap*1000

    for st_id in df.studentId.unique():
        df_student = df[df.studentId == st_id]
        start_list = df_student['startTime'].to_list()[1:]
        end_list = df_student['startTime'].to_list()[:-1]
        diff = [a_i - b_i for a_i, b_i in zip(start_list, end_list)]
        session_sub_list = [0 if i < session_gap else 1 for i in diff]
        session_sub_list.insert(0,0)
        session_sub_list = np.cumsum(session_sub_list).tolist()
        session_list.extend(session_sub_list)

    df['session_no'] = session_list
    df['session_no'] = df['session_no']+1

    df['question_no'] = df.groupby(['studentId','problemId']).cumcount()+1
    df['skill_no'] = df.groupby(['studentId','skill']).cumcount()+1

    df = df.sort_values(by=['studentId', 'startTime'])
    df = df.reset_index(drop=True)
    return df


def select_student(df, ses_min_no):
	student_use = []
	for st_id in df.studentId.unique():
		df_student = df[df.studentId == st_id]
		if max(df_student.session_no.to_list()) >=ses_min_no:
			student_use.append(st_id)

	df = df[df['studentId'].isin(student_use)]
	# df = df.sort_values(by=['studentId', 'startTime'])
	df = df.reset_index(drop=True)

	return df

def main():
	''' The main function of the preprocessing script
	'''
	parser = argparse.ArgumentParser(description='Script for preprocess')

	parser.add_argument('--dataset', type=str, default='2012',
						help='Dataset Name')

	params = parser.parse_args()
	dataset_name = params.dataset

	if dataset == '2012':
		# Load the data
		df_2012 = read_2012()

		# Featurization skill column
		df_2012 = factorize_skill_problem(df_2012)

		# Create session no for each dataset
		df_2012 = session_division(df_2012, hour=10, minute=0, second=0)
		df_2012 = select_student(df=df_2012, ses_min_no=5)

		# Output full df for Pre-train
		df_2012.to_csv('./dataset/2012.csv', index=False)

	elif dataset == '2017':
		df_2017 = read_2017()
		df_2017 = factorize_skill_problem(df_2017)
		df_2017 = session_division(df_2017, hour=10, minute=0, second=0)
		df_2017.to_csv('./dataset/2017.csv', index=False)

	elif dataset == 'Junyi':
		df_Junyi = read_Junyi()
		df_Junyi = factorize_skill_problem(df_Junyi)
		df_Junyi = session_division(df_Junyi, hour=10, minute=0, second=0)
		df_Junyi.to_csv('./dataset/Junyi.csv', index=False)

	elif dataset == 'ednet':
		df_ednet = read_ednet()
		df_ednet = factorize_skill_problem(df_ednet)
		df_ednet = session_division(df_ednet, hour=10, minute=0, second=0, ms_state=True)
		df_ednet = select_student(df=df_ednet, ses_min_no=5)
		df_ednet.to_csv('./dataset/ednet.csv', index=False)



if __name__ == '__main__':
    main()

