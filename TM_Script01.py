# Objective
	## 1. Flatten the comment data in case in comes in the following format:

{
  "Video #1": {
    "title": "Pseudo title",
    "views": "X XXX XXX",
    "up_votes": X,
    "down_votes": X,
    "Ratio_of_upvotes_to_downvotes": X,
    "author": "Pseudo creator",
    "author_subscriber": X,
    "categories": [
      "Category #1",
      "Category #2",
      "Category #3"
    ],
    "production": "Type of production",
    "description": "Description of the video",
    "duration": X,
    "upload_date": "Pseudo upload date",
    "thumbnail_url": "https://pseudo-url.jpg",
    "number_of_comment": X,
    "comments": [
      {
        "avatar": "https://pseudo-url.png",
        "username": "Pseudo username #1",
        "date": "Date of the comment #1",
        "message": "Pseudo comment #1"
      },
      {
        "avatar": "https://pseudo-url.png",
        "username": "Pseudo username #2",
        "date": "Date of the comment #2",
        "message": "Pseudo comment #2"
      },
      {
        "avatar": "https://pseudo-url.png",
        "username": "Pseudo username #3",
        "date": "Date of the comment #3",
        "message": "Pseudo comment #3"
      }
  }
  "Video #2": ...
}

	## 2. Flag non-English comments


# Import the data and modules
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import numpy as np
import pandas as pd
import itertools

## Import the data
data = pd.read_json("CleanDataComments.json")
sub_data = data


# Duplicate the rows Ni times (where N = nb of comments for the video i)
## Determine the number of times to repeat each row
com_per_vid = []

for video in range(len(sub_data)):
    com_per_vid.append(len(sub_data["comments"][video]["clean"]) + len(sub_data["comments"][video]["dirty"]))

sub_data["repeat_n_times"] = com_per_vid

## Repeat the rows
sub_data_max = sub_data.loc[sub_data.index.repeat(sub_data.repeat_n_times)].reset_index(drop=True)

## Create a list with all of the comments from all the videos
com_list = []
com_clean = []
com_total_vote = []

for video in range(len(sub_data)):
    for com in range(len(sub_data["comments"][video]["clean"])):
        comment = list(sub_data["comments"][video]["clean"][com].values())
        com_list.append(comment[3])

        cleanliness = True
        com_clean.append(cleanliness)
        
        com_total_vote.append(comment[4])
        
    for com in range(len(sub_data["comments"][video]["dirty"])):
        comment = list(sub_data["comments"][video]["dirty"][com].values())
        com_list.append(comment[3])

        cleanliness = False
        com_clean.append(cleanliness)
        
        com_total_vote.append(comment[4])
		
## Create a columns for comments ("comms") and their bot classification ("clean")
sub_data_max["comms"] = com_list
sub_data_max["clean"] = com_clean
sub_data_max["total_vote"] = com_total_vote


# Flag non-english comments
is_eng = []
for video in range(len(sub_data_max)):
    if sub_data_max["comms"][video].isascii() == True:
        eng = True
        is_eng.append(eng)
    else:
        eng = False
        is_eng.append(eng)
		
sub_data_max["is_eng"] = is_eng


# Save the data
	#Get rid of unecessary columns
sub_data_max = sub_data_max.drop(["comments", "related_videos"], axis = 1)

	#Save to csv
sub_data_max.to_csv(r'ReadyForAnalysis.csv')



