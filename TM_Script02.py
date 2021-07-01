# Objective
	## 1. Flag bots
	

# Import modules
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import pandas as pd
import numpy as np
import re


# Import data
	# Import the data
data = pd.read_json("data.json")


	# Remove rows with missing values
data = data.dropna(subset=["title"])
data = data.reset_index(drop=True)


# Cleaning
## Clean comments function
def isCleanMessage(text):

    # Detect bots
    if str(re.search(r'HTTP', text)) != "None":
        return False;

    if str(re.search(r'LIKE IF', text)) != "None":
        return False;

    if str(re.search(r'BROWSE', text)) != "None":
        return False;

    
    if str(re.search(r'MY PAGE', text)) != "None":
        return False;

    if str(re.search(r'MY CHANNEL', text)) != "None":
        return False;

    if str(re.search(r'SNAP', text)) != "None":
        return False;

    if str(re.search(r'MY PROFILE', text)) != "None":
        return False;
    
    if str(re.search(r'LISTEN UP SOLDIERS', text)) != "None":
        return False;

    # print("Clean message: ",text)
    return True


def sortComments(comments):
    clean = []
    dirty = []
    # print(co)
    for comment in comments:
        # tmp = comment['message'].upper() # if you want to keep the original message without uppercase but then need to make it uppercase
        # if (isCleanMessage(tmp])):
        comment['message'] = comment['message'].upper()
        if (isCleanMessage(comment['message'])):
            clean.append(comment)
        else:
            dirty.append(comment)

    return clean, dirty

def replaceWithCleanComments(row):
    clean, dirty = sortComments(row['comments'])
    # row['comments'] = []
    row['comments'] = {'clean': clean, 'dirty': dirty}
    # row['clean_comments'] = clean
    # row['clean_comments'] = []
    # row['dirty_comments'] = dirty

	# Aggregate function
def cleanDataFrame(dataframe):
    for i, r in dataframe.iterrows():
        replaceWithCleanComments(r)
		
## Apply the function on the data
cleanDataFrame(data)


# Save the file as Json
data.to_json(r'CleanDataComments.json')