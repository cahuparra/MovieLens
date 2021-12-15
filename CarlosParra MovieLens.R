## ----dataset_generation,  message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------------------------
# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")
if(!require(lubridate)) install.packages("lubridate", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)
library(lubridate)

# MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

# if using R 3.6 or earlier:
#movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
#                                           title = as.character(title),
#                                           genres = as.character(genres))
# if using R 4.0 or later:
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))


movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data
set.seed(1, sample.kind="Rounding")


test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set
validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set
removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)


## ----generate_datasets, message=FALSE, warning=FALSE-------------------------------------------------------------------------------------------------------------
# split edc into test and tratin datasets
test_index <- createDataPartition(edx$rating, times = 1, p = 0.5, list =  FALSE)
train_set <- edx %>% slice(-test_index)
test_set <- edx %>% slice(test_index)



## ----calculate_day, message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------------------------------
# generate predictor day
train_set<- train_set %>% 
  mutate(day = round_date(as_datetime(timestamp), unit = "day"))


## ----size_predictors, message=FALSE, warning=FALSE---------------------------------------------------------------------------------------------------------------

# Identify unique number of predictors: movie, user, time (days)

train_set %>% 
  summarize(n_users = n_distinct(userId),
              n_movies = n_distinct(movieId),
            n_days = n_distinct(day))


## ----predictor_avgrate,  message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------------------
# graphic avg_rate versus predictors
train_set %>% 
  group_by(movieId) %>% 
  summarize(avg_rate=mean(rating)) %>%
  arrange(desc(avg_rate)) %>%
  ggplot(aes(movieId,avg_rate)) + 
  geom_point() +
  geom_smooth() +
  ggtitle("Ratings by Movie Id")+
  labs(subtitle = "average ratings",
       caption = "source data : train set")
train_set %>% 
  group_by(userId) %>% 
  summarize(avg_rate=mean(rating)) %>%
  arrange(desc(avg_rate)) %>%
  ggplot(aes(userId,avg_rate)) + 
  geom_point() +
  geom_smooth() +
  ggtitle("Rating by User Id")+
  labs(subtitle = "average ratings",
       caption = "source data : train set")
#Calculate User rating fequency per day
f_ut<-train_set %>% 
  group_by(userId) %>%
  group_by(day) %>%
  summarize(f_ut=abs(log(n(),base=10000)))
f_ut %>%
  ggplot(aes(day,f_ut)) + 
  geom_point() + 
  geom_smooth() +
  ggtitle("Rating Frequency per day")+
  labs(subtitle = "average ratings",
       caption = "source data : train set")
train_set %>% 
  group_by(day) %>% 
  summarize(avg_rate=mean(rating)) %>%
  arrange(desc(avg_rate)) %>%
  ggplot(aes(day,avg_rate)) +
  geom_point()+
  geom_smooth() +
  ggtitle("Rating per day")+
  labs(subtitle = "average ratings",
  caption = "source data : train set")
  


## ----rmse_function, message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------------------------------
#function that computes the Residual Means Squared Error for a vector 
# of ratings and their corresponding predictors
RMSE <- function(true_ratings, predicted_ratings){                                         
  sqrt(mean((true_ratings - predicted_ratings)^2))
}


## ----estimate_Mu,  message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------------------------
# Estimate RMSE using Mu (average rating of all movies across all users)
mu<-mean(train_set$rating)
predicted_ratings<-mu
test_set <- test_set %>% 
     semi_join(train_set, by = "movieId") %>%
     semi_join(train_set, by = "userId")
rmse1<-RMSE(test_set$rating,predicted_ratings)
cat("Baseline/average rating across all movies and users (mu):",mu)
difference<-0
rmse_results <- tibble(method = "Baseline/average:", RMSE = rmse1, Difference = difference) 
rmse_results %>% knitr::kable() 



## ----estimate_bi,  message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------------------------
# Estimate RMSE using b_i and mu
b_i <- train_set %>% 
     group_by(movieId) %>% 
     summarize(b_i = mean(rating - mu)) 

b_i%>% ggplot(aes(b_i)) + 
  geom_histogram( bins=30,color="gray") +
  ggtitle("Histogram average ratings for movie (b_i)") +
  labs(subtitle  ="", 
       x="b_i" , 
       y="number of ratings", 
       caption ="source data : train set")


predicted_ratings<-test_set %>%
  left_join(b_i, by = "movieId") %>%
  mutate(pred =  b_i + mu) %>%
    .$pred
  
rmse2<-RMSE(test_set$rating,predicted_ratings)
difference<-rmse2-rmse1
rmse_results <-bind_rows(rmse_results,tibble(method = "Movie Effect Model", RMSE = rmse2, Difference = difference) )
rmse_results %>% knitr::kable()  



## ----estimate_bu,  message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------------------------
# Estimate RMSE using b_i and b_u
b_u <- train_set %>% 
     left_join(b_i, by='movieId') %>%
     group_by(userId) %>%
     summarize(b_u = mean(rating - mu - b_i))

b_u%>% ggplot(aes(b_u)) + 
  geom_histogram( bins=30, color = "gray") +
  ggtitle("Histogram user specific effect (b_u)") +
  labs(subtitle  ="", 
       x="b_u" , 
       y="number of ratings", 
       caption ="source data : train set")


predicted_ratings<-test_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    mutate(pred = mu + b_i + b_u) %>%
    .$pred
rmse3<-RMSE(test_set$rating,predicted_ratings)
difference<-rmse3-rmse2
rmse_results <-bind_rows(rmse_results,tibble(method = "Movie and User Effect Model", RMSE = rmse3, Difference = difference) )
rmse_results %>% knitr::kable()    


## ----estimate_fut,  message=FALSE, warning=FALSE-----------------------------------------------------------------------------------------------------------------
# Estimate RMSE using b_i andjusted by f_ut. and b_u
b_i <- train_set %>%
  left_join(f_ut,by="day") %>%
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu)/(1 - mean(f_ut )/1000)) 
b_u <- train_set %>% 
  left_join(b_i, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu - b_i))
predicted_ratings<-test_set %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
rmse4<-RMSE(test_set$rating,predicted_ratings)
difference<-rmse4-rmse3
rmse_results <-bind_rows(rmse_results,tibble(method = "Movie adjusted by frequency and  User Model", RMSE = rmse4, Difference = difference) )
rmse_results %>% knitr::kable()    


## ----estimate_bt,  message=FALSE, warning=FALSE------------------------------------------------------------------------------------------------------------------
# Estimate RMSE using b_t, b_u, b_i and mu
b_i <- train_set %>% 
     group_by(movieId) %>% 
     summarize(b_i = mean(rating - mu)) 
b_u <- train_set %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - b_i - mu))
# add day variable to tes_set to generate b_t
test_set<- test_set %>% 
  mutate(day = round_date(as_datetime(timestamp), unit = "day"))
b_t <- train_set %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(day) %>%
  summarize(b_t = mean(rating - mu - b_i - b_u))
b_t%>% ggplot(aes(b_t)) + 
  geom_histogram( bins=30, color = "gray") +
  ggtitle("Histogram time specific effect (b_t)") +
  labs(subtitle  ="", 
       x="b_t" , 
       y="number of ratings", 
       caption ="source data : train set")

predicted_ratings <-  test_set %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_t, by = "day") %>%
  mutate(pred = ifelse(is.na(mu + b_i + b_u + b_t),mu + b_i +b_u,mu + b_i + b_u + b_t   )) %>%
  .$pred
rmse5 <- RMSE(predicted_ratings, test_set$rating)
difference<-rmse5-rmse3
rmse_results <-bind_rows(rmse_results,tibble(method = "Movie,User, and Time Model", RMSE = rmse5, Difference = difference) )
rmse_results %>% knitr::kable()  


## ----estimate_lambda,  message=FALSE, warning=FALSE--------------------------------------------------------------------------------------------------------------
lambdas <- seq(0,10, 0.25)
rmses <- sapply(lambdas, function(l){
  b_i <- train_set %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
 
  b_u <- train_set %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_t <- train_set %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(day) %>%
    summarize(b_t = sum(rating - mu - b_i - b_u)/(n()+l))
  predicted_ratings <- 
    test_set %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = "day") %>%
    mutate(pred = ifelse(is.na(mu + b_i + b_u + b_t),mu + b_i +b_u,mu + b_i + b_u + b_t)) %>%
    .$pred
  return(RMSE(predicted_ratings, test_set$rating))
})
qplot(lambdas, rmses) 
lambda <- lambdas[which.min(rmses)]
cat("Optimal Lambda:", lambda)
rmse6<- min(rmses)
difference<-rmse6-rmse5
rmse_results <-bind_rows(rmse_results,tibble(method = "Regularized Movie, User, and and Time Model", RMSE = rmse6, Difference = difference) )
rmse_results %>% knitr::kable() 



## ----validation,  message=FALSE, warning=FALSE-------------------------------------------------------------------------------------------------------------------
edx <- edx %>% 
    mutate(day = round_date(as_datetime(timestamp), unit = "day"))  

validation <- validation %>% 
  mutate(day = round_date(as_datetime(timestamp), unit = "day"))
 
b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+lambda))
 
b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
 
 b_t <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(day) %>%
    summarize(b_t = sum(rating - mu - b_i - b_u)/(n()+lambda))
 predicted_ratings <- 
    validation %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = "day") %>%
    mutate(pred = ifelse(is.na(mu + b_i + b_u + b_t),mu + b_i +b_u,mu + b_i + b_u + b_t   )) %>%
    .$pred
rmse7<-RMSE(validation$rating,predicted_ratings)
difference<-rmse7-rmse6
rmse_results <-bind_rows(rmse_results,tibble(method = "Validation - Regularized Movie, User, and and Time Model", RMSE = rmse7, Difference = difference) )
rmse_results %>% knitr::kable() 



## ----result, message=FALSE, warning=FALSE, echo=FALSE------------------------------------------------------------------------------------------------------------
cat("OPTIMAL RMSE:", rmse7)

