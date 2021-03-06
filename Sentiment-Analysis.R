rm(list = ls())

# Load the data and libraries ---------------------------------------------

library(tidyverse)
library(tm)
library(qdap)
library(SnowballC)

data <- read.csv("ReadyForAnalysis.csv", header = TRUE)
data <- subset(data, data$clean == "True" & data$is_eng == "True")

columns <- colnames(data)
columns

# Create cleaning function ------------------------------------------------

CleaningFunction <- function(text){
  
  # Convert to lowercase
  text <- tolower(text)
  
  # Replace abbreviations
  replace_abbreviation(text)
  
  # Replace contractions
  replace_contraction(text)
  
  # Replace symbols with words
  replace_symbol(text)
  
  # Remove numbers
  text <- removeNumbers(text)
  
  # Add custom stopwords
  extra_stops <- c('awww', 
                   'afternoon', 
                   'mmmm', 
                   'haha',
                   'that',
                   'xoxo',
                   'soooo',
                   'mmmmm',
                   'gonna',
                   'bruh',
                   'yeah',
                   'sooo',
                   'yummi',
                   "shes",
                   "yooou",
                   "mmmmmmm",
                   "yoou",
                   "xoxoxo",
                   "sooooo",
                   "yaay",
                   "ohhh",
                   "hehe",
                   "xxxx",
                   "xoxo",
                   "soooooo",
                   "mmmmmm",
                   "hahaha",
                   "xxxxx",
                   "will",
                   "couldn",
                   "gostosa",
                   "deliciosa",
                   "delicia",
                   "instead",
                   "delicioso",
                   "esta",
                   "como",
                   "boca",
                   "buen",
                   "est",
                   "quiero",
                   "youu",
                   "wanna",
                   "yesss",
                   "princess",
                   "bella",
                   "aussi",
                   "yessss",
                   "hmmm",
                   "hermosa",
                   "para",
                   "pero",
                   "hermoso",
                   "wasn",
                   "oooh",
                   "sehr",
                   "geil",
                   "yall")
  
  
  # Remove common English stopwords
  text <- removeWords(text, c(extra_stops, stopwords("en")))
  
  # Remove unecessary whitespaces
  text <- stripWhitespace(text)
  
  # Remove special characters
  text <- gsub("[[:punct:]]", "", text)
  
  # Return cleaned text
  return(text)
}


# Clean the comments ------------------------------
data$cleaned_comments <- CleaningFunction(data$comms)


# Remove empty (non-informative) comments ---------------------------------
data <- subset(data, data$cleaned_comments != "")


# Apply sentiment analysis to final dataframe ------------------------------------------------------

# Customise sentiment analysis dictionary 
positives <- c(positive.words, "queen","suck", "hardcore", "god", "tits", "destroy", 
               "bust", "amazing", "hard", "dick", "nice", "thanks", "hot", 
               "hooot", "amazzing", "sexy", "princess", "sexiest", "amazing", "awesome", 
               "gorgeous", "fantastic", "beautiful", "perfect", "favorite", "sweet",
               "best", "super", "incredible", "damn", "hotttt","whoa", "dang", "woow",
               "sexiest", "turn on", "love", "goddess", "gawdamn", "niceee", "relationship goals")
negatives <- c(negative.words, "turn off", "bummer", "minuscule", "tiny",
               "creep", "creepy", "creeper")

negatives <- negatives[negatives != "fuck"]
negatives <- negatives[negatives != "fucking"]

negation.words <- c(negation.words, "dont", "didnt")

amplification.words <- c(amplification.words, "fucking", "fuck", "fuckkk")


new_lexicon <- sentiment_frame(positives, negatives, pos.weights = 1, neg.weights = -1)


# Create polarity conversion function
ConvertToLevel <- function(polarity){
  if (polarity > 0){
    return(1)
  }
  
  if (polarity == 0){
    return(0)
  }
  
  if (polarity < 0){
    return(-1)
  }
}


# Compute polarity score
pol <- polarity(data$cleaned_comments)
data$polarity <- pol$all[,3]


# Convert polarity score to 3 class
data$polarity <- sapply(data$polarity, ConvertToLevel)
data$polarity <- as.factor(data$polarity)


# Save the data
write.csv(data, "Final_data_processed.csv", row.names = FALSE)
