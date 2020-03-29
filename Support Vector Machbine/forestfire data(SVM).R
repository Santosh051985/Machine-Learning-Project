#####Support Vector Machines -------------------
##  Optical Character Recognition ----
#load salary data set

forestfires<-read.csv(file.choose())
View(forestfires)

hist(forestfires$area)
rug(forestfires$area)

normalise <- function(x) {
  return((x - min(x)) / (max(x) - min(x)))  # subtract the min value in x and divide by the range of values in x.
}

forestfires$temp <- normalise(forestfires$temp)
forestfires$rain <- normalise(forestfires$rain)
forestfires$RH <- normalise(forestfires$RH)
forestfires$wind <- normalise(forestfires$wind)

sum(forestfires$area < 5) 
sum(forestfires$area >= 5)

forestfires_train<-data.frame(forestfires[1:413, ]
forestfires_test<-data.frame(forestfires[414:517, ]

##Training a model on the data ----
# begin by training a simple linear SVM
install.packages("kernlab")
library(kernlab)

forestfires_classifier <- ksvm(size_category ~ ., data = forestfires_train,
                          kernel = "vanilladot")

?ksvm
# basic information about the model
forestfires_classifier

## Evaluating model performance ----
# predictions on testing dataset
forestfires_predictions <- predict(forestfires_classifier, forestfires_test)

head(forestfires_predictions)
table(forestfires_predictions, forestfires_test$size_category)

agreement <- forestfires_predictions == forestfires_test$size_category
table(agreement)
prop.table(table(agreement))

## Improving model performance ----
forestfires_classifier_rbf <- ksvm(size_category ~ ., data = forestfires_train, kernel = "rbfdot")
forestfires_predictions_rbf <- predict(forestfires_classifier_rbf, forestfires_test)

agreement_rbf <- forestfires_predictions_rbf == forestfires_test$size_category
table(agreement_rbf)
prop.table(table(agreement_rbf))
?ksvm

# here after building the rbf model there is no accauracy in the data set such that the given first model is good because it has got the 98% accuaracy

m.poly <- ksvm(size_category~.,
               data = forestfires_train,
               kernel = "polydot", C = 1)
m.ploy_prediction<-predict(m.poly,forestfires_test)
agreement_m.poly<-m.ploy_prediction==forestfires_test$size_category
table(agreement_m.poly)
prop.table(table(agreement_m.poly))

# here the polynominal model has good accuaracy 99%

tanhdot <- ksvm(size_category~.,
               data = forestfires_train,
               kernel = "tanhdot", C = 1)
tanhdot_prediction<-predict(tanhdot,forestfires_test)
agreement_tanhdot<-tanhdot_prediction==forestfires_test$size_category
table(agreement_tanhdot)
prop.table(table(agreement_tanhdot))
