##########################################################
# Create edx set, validation set (final hold-out test set)
##########################################################

# Note: this process could take a couple of minutes

if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

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
set.seed(1, sample.kind="Rounding") # if using R 3.5 or earlier, use `set.seed(1)`
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

#function that computes the Residual Means Squared Error for a vector of ratings and their corresponding predictors
RMSE <- function(true_ratings, predicted_ratings){                                         
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#Use variable timestamp to generate predictor week
edx<- edx %>% 
  mutate(week = round_date(as_datetime(timestamp), unit = "week"))
# Identify unique number of predictors (movie, user, time (weeks))
edx %>% 
  summarize(n_users = n_distinct(userId),
              n_movies = n_distinct(movieId),
            n_weeks = n_distinct(week))

#Normalization of Effects using a model based approach
# Equation: Y_ui=mu+bi+bu+b_t epsilon_ui
#    Estimate mu (average rating of all movies accross all users)
mu<-mean(edx$rating)
#   Estimate b_i (average rating for movie i), b_u (user specific effect), and b_t (time effect)
#   Regularize for movie, user, and effect
#     tune lambda symmetrically for bi and bu and bt to minimize RMSE by using cross-validation
lambdas <- seq(0, 5, 0.25)
rmses <- sapply(lambdas, function(l){
  b_i <- edx %>%
    group_by(movieId) %>%
    summarize(b_i = sum(rating - mu)/(n()+l))
  b_u <- edx %>% 
    left_join(b_i, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_u = sum(rating - b_i - mu)/(n()+l))
  
  b_t <- edx %>%
    left_join(b_i, by='movieId') %>%
    left_join(b_u, by='userId') %>%
    group_by(week) %>%
    summarize(b_t = sum(rating - mu - b_i - b_u)/(n()+l))
  predicted_ratings <- 
    edx %>% 
    left_join(b_i, by = "movieId") %>%
    left_join(b_u, by = "userId") %>%
    left_join(b_t, by = "week") %>%
    mutate(pred = mu + b_i + b_u + b_t) %>%
    .$pred
  return(RMSE(predicted_ratings, edx$rating))
})

qplot(lambdas, rmses)  
lambda <- lambdas[which.min(rmses)]
min(rmses)    #0.8565926
predicted_ratings
nrow(b_t)
nrow(f_ui)
f_ui
f_ui<-edx %>% 
  left_join(b_u, by='userId') %>%
  group_by(week) %>%
  summarize(f_ui=log(n()/7))
  
  
#Execute Regularization with optimized lambda
b_i <- edx %>%
   group_by(movieId) %>%
  summarize(b_i = (sum(rating - mu)/(n()+lambda)))
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
b_t <- edx %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(week) %>%
  summarize(b_t = sum(rating - mu - b_i - b_u)/(n()+lambda))
predicted_ratings <- 
  edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_t, by = "week") %>%
   mutate(pred = mu + b_i + b_u +b_t ) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, edx$rating)
model_3_rmse   #0.8565926

#______________________
# Vn for users and Ui for movies


freq_usr <- edx %>% group_by(week) %>%
     select(week, userId,  rating) %>%
     group_by(week,userId) %>% summarize(n=n(),avg_rat=mean(rating))
dim(freq_usr)


# DATA HISTOGRAMS
# histogram movie effect b_i 

edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)) %>%       
  ggplot(aes(b_i)) + 
  geom_histogram(bins = 30, color = "blue")
# histogram  user effect b_u
edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)) %>%
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black",fill="green")

#histogram movie effect b_i with regularization
edx %>% 
  group_by(movieId) %>% 
  summarize(b_i = sum(rating - mu)/(n()+lambda)) %>%       
  ggplot(aes(b_i)) + 
  geom_histogram(bins = 30, color = "blue") 
# histogram  user effect b_u with regularization
edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda)) %>%
    ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black",fill="green")
#histogram time effect b_t with regularization
edx %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  group_by(week) %>%
  summarize(b_t = sum(rating - mu - b_i - b_u)/(n()+lambda)) %>%
  ggplot(aes(b_t)) + 
  geom_histogram(bins = 30, color = "black",fill="blue")


edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_t, by = "week") %>%
  mutate(pred = mu + b_i + b_u +b_t)%>%  
  ggplot(aes(rating,pred)) + geom_point()
  
  geom_histogram(bins = 30, color = "black",fill="black")





edx  %>% ggplot(aes(rating)) +geom_histogram(bins=30,fill="black")

str(predicted_ratings)
# histogram average rating for user u (user effect)
edx %>% 
  group_by(userId) %>% 
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, colour = "red") 

#graphic representing the rating trend through weeks
edx %>% group_by(week) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(week, rating)) +
  geom_point() +
  geom_smooth()

#graphic representing the effect of the user rating frequency on the average rating
freq_usr %>% group_by(userId) %>% filter(n()>50) %>%
  ggplot(aes(log(n/7),avg_rat))+geom_point()+geom_smooth()

freq_usr %>% group_by(userId) %>% filter(n()>1000) %>%
  ggplot(aes(userId,avg_rat,size=log(n/7)))+geom_point()+geom_smooth()

freq_usr
max(freq_usr$n)

f_y <- edx %>% 
  group_by(week) %>%
  summarize(f_y=n(), user=userId)


nrow(f_y)
f_y

# generate a userId, Title matrix with ratings 
# consider only movies with more than 527 ratings (90%) and users with more than 49 reviews (90%)
movie_titles <- edx %>%                                                                 # select database that includes movieId and title
  select(movieId, title) %>%
  distinct()
train_small <- edx %>% 
  group_by(movieId) %>% 
  filter(n() >= 527 ) %>% ungroup() %>% 
  group_by(userId) %>%
  filter(n() >= 49) %>% ungroup()

y <- train_small %>% 
  select(userId, movieId, rating) %>%
  spread(movieId, rating) %>%
  as.matrix()

dim(y)
y[1:10,1:2]
# the userID is defined as the row name
rownames(y)<- y[,1]
y <- y[,-1]
# the movie title is assigned as the column name
colnames(y) <- with(movie_titles, title[match(colnames(y), movieId)])
# replace NA with zero
y[is.na(y)] <- 0
# convert these residuals by removing the column and row averages
#residuals are not independent of each other
y <- sweep(y, 1, rowMeans(y, na.rm=TRUE))                    
y <- sweep(y, 2, colMeans(y, na.rm=TRUE))                      

prc<-prcomp(y)
str(prc)
dim(prc$rotation)  
qplot(1:nrow(prc$x[,1]), prc$sdev, xlab = "PC")
prc$x[2]

# principal components q. movie effect  (545,292)
plot(summary(prc)$importance[3,])
boxplot(prc$x[,7] ~ colnames(y), main = paste("PC", 7))

dim(prc$x)
prc$x[,7]
prc$x[1:5,1:5]
y[1:5,1:5]
m_1 <- "Grumpier Old Men (1995)"
m_2 <- "Waiting to Exhale (1995)"
p1 <- qplot(y[ ,m_1], y[,m_2], xlab = m_1, ylab = m_2)
p1

install.packages('ggrepel', repos = 'https://cloud.r-project.org')
library(ggrepel)
pcs <- data.frame(prc$rotation, name = colnames(y))
pcs %>%  ggplot(aes(PC1, PC2)) + geom_point() + 
  geom_text_repel(aes(PC1, PC2, label=name),
                  data = filter(pcs, 
                                PC1 < -0.1 | PC1 > 0.1 | PC2 < -0.075 | PC2 > 0.1))
pcs %>% select(name, PC1) %>% arrange(PC1) %>% slice(1:10)
pcs %>% select(name, PC1) %>% arrange(desc(PC1)) %>% slice(1:10)
pcs %>% select(name, PC2) %>% arrange(PC2) %>% slice(1:10)
pcs %>% select(name, PC2) %>% arrange(desc(PC2)) %>% slice(1:10)

scaled_y<-scale(y)
svd<- svd(y,na.omit)

scaled_y[,1]
?scale
str(y)
prc<-prcomp(y)
?prcomp





sum(movies$count[1:2770])/sum(movies$count)
movies$count[2770]
users<-edx %>% 
  group_by(userId) %>%
  summarize(count=n()) %>%
  arrange(desc(count))
users
nrow(users)
library(matrixStats)
sum(users$count[1:41000])/sum(users$count)
users$count[41000]





predicted_ratings <- edx %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu + b_i + b_u) %>%
  .$pred
model_2_rmse <- RMSE(predicted_ratings, edx$rating)
model_2_rmse   #0.8567
predicted_ratings <- mu + edx %>% 
  left_join(movie_avgs, by='movieId') %>%        #included bi in the prediction
  .$b_i
model_1_rmse <- RMSE(predicted_ratings, edx$rating)
model_1_rmse    #0.9423


s<-svd(y)
s$d[1:50]
dim(s$u)
svdratings1<- s$u %*% s$d %*% s$v
s$u
svdratings<- svdratings1 %*% t(s$v)
str(s)
plot(s$u[,1], ylim = c(-0.25, 0.25))

nrow(f_ui)
b_i[2]
f_ui[2]
log(f_ui[2])

b_i1 <- edx %>%
  group_by(movieId)
?log
f_ui<-edx %>% 
  left_join(b_u, by='userId') %>%
  group_by(week) %>%
  summarize(f_ui=log(n()/7))
b_t

#Execute Regularization with optimized lambda
b_i <- edx %>%
  group_by(movieId) %>%
  summarize(b_i = (sum(rating - mu)/(n()+lambda)))
b_u <- edx %>% 
  left_join(b_i, by="movieId") %>%
  group_by(userId) %>%
  summarize(b_u = sum(rating - b_i - mu)/(n()+lambda))
b_t <- edx %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(f_ui, by= 'week') %>%
  group_by(week) %>%
  summarize(b_t = sum(rating - mu - b_i -b_i*f_ui - b_u)/(n()+lambda))
f_ui<-edx %>% 
  left_join(b_u, by='userId') %>%
  group_by(week) %>%
  summarize(f_ui=log(n()/7))
predicted_ratings <- 
  edx %>% 
  left_join(b_i, by = "movieId") %>%
  left_join(b_u, by = "userId") %>%
  left_join(b_t, by = "week") %>%
  mutate(pred = mu + b_i + b_u +b_t ) %>%
  .$pred

model_3_rmse <- RMSE(predicted_ratings, edx$rating)
model_3_rmse   #0.8565926


edx_factorization <- edx %>% select(movieId,userId,rating)
validation_factorization <- validation %>% select(movieId,userId,rating)

edx_factorization <- as.matrix(edx_factorization)
validation_factorization <- as.matrix(validation_factorization)
write.table(edx_factorization , file = "trainingset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)



write.table(validation_factorization, file = "validationset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)
set.seed(1)
install.packages("recosystem")
library(recosystem)
training_dataset <- data_file( "trainingset.txt")

validation_dataset <- data_file( "validationset.txt")
r = Reco()
opts = r$tune(training_dataset, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                            costp_l1 = 0, costq_l1 = 0,
                                            nthread = 1, niter = 10))
r$train(training_dataset, opts = c(opts$min, nthread = 1, niter = 20))


pred_file = tempfile()

r$predict(validation_dataset, out_file(pred_file)) 
print(scan(pred_file, n = 10))
scores_real <- read.table("validationset.txt", header = FALSE, sep = " ")$V3
scores_pred <- scan(pred_file)
rm(edx.copy, valid.copy)

rmse_mf <- RMSE(scores_real,scores_pred)
rmse_mf <- RMSE(edx$rating,scores_pred)
str(training_dataset)

rmse_mf  

b_g <- split_edx %>%
  left_join(b_i, by='movieId') %>%
  left_join(b_u, by='userId') %>%
  left_join(b_y, by = 'year') %>%
  group_by(genres) %>%
  summarize(b_g = sum(rating - mu - b_i - b_u - b_y)/(n()+lambda), n_g = n())
