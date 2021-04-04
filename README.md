# who-is-the-asshole

An NLP moral philosopher that will read a text description of a conflict, and make a moral judgement on whether or not the writer is the asshole in the situation.

# Data Source
The data came from the subreddit [r/amitheasshole](reddit.com/r/AmItheAsshole/) and was scraped by this [repo](github.com/iterative/aita_dataset)

According to the subreddit itself, it's a place to tell the story of a non-violent conflict you have experienced; give both sides of the story, and find out if you're right, or you're the asshole. 

The features used in this project are 
- Post Title
- Post Body
- Verdict (YTA, NTA, ESH, NAH)
- Is Asshole

Below reference chart came from the automod in the subreddit. Each comment on a post is required to give one of below 5 judgements. The final verdict is presented as a flair on the post, which is based on the judgements given in the comments by community members. 
![verdict chart](/image/verdict.png)

Since it's not important for the purpose of this project if the other party in the story is the asshole, YTA and ESH are converted into 1, NTA and NAH are converted into 0, representing if the OP (original poster) is the asshole or is not the asshole.

# EDA
## Verdict Frequency
![](/image/verdict_bar.png)
Among the 97,455 posts in this dataset, majority of people received a verdict of NTA (not the asshole). 

This makes sense because people tend to post a story in which they are in the right, to look for validation from internet strangers. If a person knows that he/she is obviously in the wrong, then he/she is less likely to post the story knowing that they will be grilled and called out by the internet strangers. 

This created a problem of imbalanced classes for building the ML model, my solution was to upsample the minority class using SMOTE.
I chose to upsample minority class instead of downsampling majority class because I didn't want to lose information. 

## Post Scores
![](/image/post_freq_hist.png)
The score of a post is upvotes - downvotes. This is a metric of popularity, the score on a post doesn't necessarily have an implication on the verdict, since any interesting story can receives a lot of attention and upvotes regardless of the verdict. 

This dataset only took in posts that have a minimum of 3. Majority of posts in this dataset have a score around 3 - 20. 
This is normal since only minority of posts become very popular. 

## Post Length
![](/image/post_length.png)
Majority of the posts have less than 700 words. 

## Sex and Age of OP
![](/image/age_sex.png)
It is customary on reddit to state your age and sex in a format similar to the following examples:
- I am (55F)
- me [17M] 
- I'm a [24FtM]
- my (30M) wife(30F)

However, this subreddit does not require people to disclose their age and sex explicitly, some chose not to mention these information, and some mentioned these information in the context without any specific format, for example
- "My wife and I are in our 30s, ... Am I a bad husband for ..."
- "My friends and I always have our girls' night on Friday" 

In these cases a human can read between the lines and make out the sex or age of the OP, but it's difficult to extract using regex. So I did not use sex and age as predictors in my models.

# Process
1. Data cleaning round 1: remove links and lne breaks from the posts, expand contractions (I won't -> I will not).
2. Sentiment Analysis: give each post a polarity and subjectivity score using textblob
3. Data cleaning round 2: remove punctuations
4. Turn posts into document term matrix using count vectorizer, with unigrams and bigrams, and removing English stopwords
5. Combine sentiment analysis and document term matrix (DTM).
6. SMOTE - oversample minority class
7. Feed the smoted DTM + sentiment analysis into the models as predictors. 

# Models
The best performing models are below, with corss validation scores:
### 1. SGD  
    F1 Score = 0.671  
    Accuracy Score = 0.727
### 2. Random Forest
    F1 Score = 0.673  
    Accuracy Score = 0.765

# Moral Of The Story
- Nuances can make a difference in moral judgement
    - Humans can read between the lines and refuse to accept an illogical scenario described by OP 
    - OP's comments often contribute to the final verdict, comments are not included in the body
- Moral standards can be subjective
    - can we trust reddit's judgement? 
- Bag-of-words approach is inadequate to account for nuances.
    - word2vec seems to be more appropriate in retaining meanings of text

## Next Step
- word2vec

