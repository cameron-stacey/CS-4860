inputData <- read.csv("http://rstatistics.net/wp-content/uploads/2015/09/adult.csv")
head(inputData)
table(inputData$ABOVE50K)

input_ones <- inputData[which(inputData$ABOVE50K == 1), ]
input_zeros <- inputData[which(inputData$ABOVE50K == 0), ]
set.seed(100)
input_ones_training_rows <- sample(1:nrow(input_ones), 0.7*nrow(input_ones))
input_zeros_training_rows <- sample(1:nrow(input_zeros), 0.7*nrow(input_ones))
training_ones <- input_ones[input_ones_training_rows, ] 
training_zeros <- input_zeros[input_zeros_training_rows, ]
trainingData <- rbind(training_ones, training_zeros)
test_ones <- input_ones[-input_ones_training_rows, ]
test_zeros <- input_zeros[-input_zeros_training_rows, ]
testData <- rbind(test_ones, test_zeros)

library(smbinning)
factor_vars <- c ("WORKCLASS", "EDUCATION", "MARITALSTATUS", "OCCUPATION", "RELATIONSHIP", "RACE", "SEX", "NATIVECOUNTRY")
continuous_vars <- c("AGE", "FNLWGT","EDUCATIONNUM", "HOURSPERWEEK", "CAPITALGAIN", "CAPITALLOSS")
iv_df <- data.frame(VARS=c(factor_vars, continuous_vars), IV=numeric(14))

for(factor_var in factor_vars){
  smb <- smbinning.factor(trainingData, y="ABOVE50K", x=factor_var)  # WOE table
  if(class(smb) != "character"){ # heck if some error occured
    iv_df[iv_df$VARS == factor_var, "IV"] <- smb$iv
  }
}

for(continuous_var in continuous_vars){
  smb <- smbinning(trainingData, y="ABOVE50K", x=continuous_var)  # WOE table
  if(class(smb) != "character"){  # any error while calculating scores.
    iv_df[iv_df$VARS == continuous_var, "IV"] <- smb$iv
  }
}

iv_df <- iv_df[order(-iv_df$IV), ]
iv_df

logitMod <- glm(ABOVE50K ~ RELATIONSHIP + AGE + CAPITALGAIN + OCCUPATION + EDUCATIONNUM, data=trainingData, family=binomial(link="logit"))
predicted <- plogis(predict(logitMod, testData))

library(InformationValue)
optCutOff <- optimalCutoff(testData$ABOVE50K, predicted)[1]

#analyzation
summary(logitMod)
misClassError(testData$ABOVE50K, predicted, threshold = optCutOff)
plotROC(testData$ABOVE50K, predicted)
Concordance(testData$ABOVE50K, predicted)
sensitivity(testData$ABOVE50K, predicted, threshold = optCutOff)
specificity(testData$ABOVE50K, predicted, threshold = optCutOff)
confusionMatrix(testData$ABOVE50K, predicted, threshold = optCutOff)