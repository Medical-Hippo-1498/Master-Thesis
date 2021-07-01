# Title: Added Value of User-Generated Content in Online Adult Entertainment
  #### Description: 
This GitHub project was created in the context of my Master Thesis for the program of Business Intelligence and Smart Services at Maastricht University.
  
In this thesis, I fill a gap in the academic literature by deriving the added value of user-generated content (UGC) and investigating the idiosyncrasies of the online adult entertainment industry with regard to topic modelling and sentiment analysis using synthetic as well as YouTube comment data. Moreover, drawing on the significant predictive power of UGC on the popularity of online adult entertainment videos, I develop a data analytics dashboard which enhances the communication channels between content creators and users following strong ethical and legal guidelines, to answer the call for empowerment of independent creators in an inherently innovative industry that is the one of online adult entertainment. This research thus provides a societal contribution just as significant as it sheds light on a field under-represented in the academic literature. 
##
## -Author: Harris Othman
##
## -Supervisor: Asssitant Professor Dr. Leto Peel
##
### Feel free to contact me via h.othman@student.maastrichtuniversity.nl or othman.harris@gmail.com in case you have any questions with regard to this project.

#
## -Steps (Topic Modelling (TM)) (Programming language: Python):
  ### 1. Transform the data for analysis (i.e., flatten the json file based on the number of comments per video)
      #### Note: If your dataset already has an equal amount of rows for each comment per video, then you can skip this step
  ### 2. Clean the data (i.e., flag bots and non-English comments)
  ### 3. Fit a LDA model
##
## -Steps (Sentiment Analysis) (programming language: R):
  ### 1. Clean the comments
  ### 2. Create a custom sentiment analysis dictionary
  ### 3. Compute the polarity score for each of the comments
