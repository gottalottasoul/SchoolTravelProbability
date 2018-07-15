###########################################################################
# School Travel Propensity Model                                        ###
# Blake Abbenante                                                       ###
# Initial Version: 3/23/2015                                            ###
###########################################################################

#set working directory
setwd("C:\\Users\\Blake.Abbenante\\Google Drive\\Work\\r\\School Score\\")

#load the required libraries
require(caret) || install.packages("caret", repos="http://cran.rstudio.org") 
library(caret)
require(car) || install.packages("car", repos="http://cran.rstudio.org") 
library(car)
require(randomForest) || install.packages("randomForest", repos="http://cran.rstudio.org")
library(randomForest)
require(rpart) || install.packages("rpart", repos="http://cran.rstudio.org") 
library(rpart)
require(AUC) || install.packages("AUC", repos="http://cran.rstudio.org") 
library(AUC)
require(SDMTools) || install.packages("SDMTools", repos="http://cran.rstudio.org") 
library(SDMTools)
require(ROCR) || install.packages("ROCR", repos="http://cran.rstudio.org")
library(ROCR)
require(gbm) || install.packages("gbm", repos="http://cran.rstudio.org") 
library(gbm)
require(reshape) || install.packages("reshape", repos="http://cran.rstudio.org") 
library(reshape)
require(ada) || install.packages("ada", repos="http://cran.rstudio.org") 
library(ada)
require(e1071) || install.packages("e1071", repos="http://cran.rstudio.org") 
library(e1071)
require(h2o) || install.packages("h2o", repos="http://cran.rstudio.org") 
library(h2o)
require(vtreat) || install.packages(install.packages('vtreat_0.2.tar.gz',repos=NULL,type='source')) 
library(vtreat)


## clear the console of all variables
rm(list = ls())
## free up unused memory
gc()

###########################################################################
# Functions                                                             ###
###########################################################################

BinVars <- function(x){
  #  as.numeric(cut(x, 10))
  #  discretize(x,disc="equalfreq",10)
  discretize(x,disc="equalwidth",10)
}

GetCategoryCounts <- function(x) {
  nlevels(x)
}

ImputeByMean <- function(x) {
  replace(x, is.na(x), mean(x, na.rm = TRUE))
}

printDensityPlots <- function(x){
  densityplot(~x,groups=res_upsell,data=orange_train_final,plot.points=FALSE,auto.key=TRUE)
}

printSingleBoxPlot <- function(myVar,myResponse){
  boxplot(myVar~myResponse,data=orange_train_final,outline=FALSE,ylab=myVar,xlab=myVar,col=c("blue","red"))
}

my.summary <- function (data, varname)
{
  g.var <- paste(varname, "sum", sep=".")
  outcomes <- c("outcome")
  vars <- setdiff(colnames(data), c(outcomes, "rgroup"))
  numvars <- vars[sapply(data[, vars], class) %in% c("numeric", "integer")]
  k <- length(numvars)
  sum.df <- data.frame(matrix(NA, nrow = 0, ncol = 16))
  colnames(sum.df) <- c("Variable", "No.Obs", "Missing", "MissPerc","UniqueValues","Minimum", "1st.Qtr", "Median", "3rd.Qtr",
                        "Maximum", "Mean", "Std.Dev", "Variance", "Range", "Lo.Outs", "Hi.Outs")
  for (i in 1:k)
  {
    lo.out <- quantile(data[, i], probs=.25, na.rm=T) - 1.5*(quantile(data[, i], probs=.75, na.rm=T)-quantile(data[, i], probs=.25, na.rm=T))
    hi.out <- quantile(data[, i], probs=.75, na.rm=T) + 1.5*(quantile(data[, i], probs=.75, na.rm=T)-quantile(data[, i], probs=.25, na.rm=T))
    sum <- cbind(Variable=names(data[i]),
                 No.Obs=NROW(data[, i]),
                 Missing=sum(is.na(data[, i])) + sum(is.null(data[, i])),
                 MissPerc=((sum(is.na(data[, i])) + sum(is.null(data[, i])))/(NROW(data[, i]))),
                 UniqueValues=length(unique(data[,i])),
                 Minimum=quantile(data[, i], probs=0, na.rm=T),
                 "1st.Qtr"=quantile(data[, i], probs=.25, na.rm=T),
                 Median=quantile(data[, i], probs=.50, na.rm=T),
                 "3rd.Qtr"=quantile(data[, i], probs=.75, na.rm=T),
                 Maximum=quantile(data[, i], probs=1, na.rm=T),
                 Mean=round(mean(data[, i], na.rm=T),4),
                 Std.Dev=round(sd(data[, i], na.rm=T),4),
                 Variance=round(var(data[, i], na.rm=T),4),
                 Range=quantile(data[, i], probs=1, na.rm=T)-quantile(data[, i], probs=0, na.rm=T),
                 Lo.Outs=sum(isTRUE(data[, i] < lo.out[[1]])),
                 Hi.Outs=sum(isTRUE(data[, i] < hi.out[[1]])))
    sum.df[i, ] <- sum
  }
  sum.df[, 2:16] <- sapply(sum.df[, 2:16], as.numeric)
  assign(g.var, sum.df, envir = .GlobalEnv)
  return(sum.df)
}

BuildDecisionTree <- function(myResponse)
{
  fit <- rpart(myResponse ~.,method="class", data=dt_data)
  plot(fit, margin=0.1)  # pad around the plot to allow text to show
  text(fit, use.n = TRUE, all=FALSE, cex=0.8)  # cex is a scale for marker or text size
  
  printcp(fit) # display the results 
  plotcp(fit) # visualize cross-validation results 
  summary(fit) # detailed summary of splits
  
  return(fit)
}

BuildContingencyTables <- function(myVar, myResponse)
{
  varTable <- table(myVar=myVar,responsevar=myResponse,useNA='ifany')
  write.table(varTable, file=paste("c:\\RStuff\\table",colnames(myVar),".txt"), sep="\t")  
}

#FUNCTION TO BUILD SINGLE VAR MODELS - categorical
mkPredC <- function(outCol,varCol,appCol) {
  pPos <- sum(outCol==pos)/length(outCol) #how often outcome is +1 in train
  naTab <- table(as.factor(outCol[is.na(varCol)])) #how often outcome is +1 for NA
  pPosWna <- (naTab/sum(naTab))[pos] 
  vTab <- table(as.factor(outCol),varCol) 
  pPosWv <- (vTab[pos,]+1.0e-3*pPos)/(colSums(vTab)+1.0e-3) #how often +1 conditioned on lvls of training var
  pred <- pPosWv[appCol]  #make predictions
  pred[is.na(appCol)] <- pPosWna #add predictions for NA lvls
  pred[is.na(pred)] <- pPos #add predictions for lvls of appCol that weren't known in training
  pred
}

#scoring categorical var by AUC
calcAUC <- function(predcol,outcol) {
  perf <- performance(prediction(predcol,outcol==pos),'auc')
  as.numeric(perf@y.values)
}

###########################################################################
# End Functions                                                         ###
###########################################################################


school_data.full=read.csv("data\\school_data.csv",colClasses=c(rep("factor",8),rep("numeric",44),rep("factor",45)))

#Create var for first 3 of zipcode
#school_data.traveled<-school_data.full[which(school_data.full$ACPPaxInLastTwoYears>0 | school_data.full$Prev6YrPPax>0),]
school_data.traveled<-school_data.full
school_data.traveled$first3zip<-substring(school_data.traveled$ZipCode,1,3)
school_data.traveled$first3zip <- factor(school_data.traveled$first3zip)


#Code numeric and categorical variables
#outcomes=c('TraveledInLastTwoYears')
vars <- setdiff(colnames(school_data.traveled),c('rgroup'))

catVars <- vars[sapply(school_data.traveled[,vars],class) %in% c('factor','character')]
numericVars <- vars[sapply(school_data.traveled[,vars],class) %in% c('numeric','integer')]
logicalVars <- vars[sapply(school_data.traveled[,vars],class) %in% c('logical')]



school_data.traveled.numeric.sum <- data.frame(my.summary(school_data.traveled[numericVars],"School Data"))
#look at distribution of missing values
ggplot(data = school_data.traveled.numeric.sum, aes(x = MissPerc)) + scale_x_continuous(name = "Percent Values NA") + scale_y_continuous(name="Number of Predictor Vars") +  geom_bar(binwidth = 0.05, fill = "firebrick4", color = "dodgerblue2") + ggtitle("Distribution of Missing Values on All Predictors") + theme(plot.title = element_text(lineheight=.8, face="bold"))

#get some summary data for categorical predictors, and stick them in a DF
school_data.traveled.categorical.sum <- sapply(school_data.traveled[catVars],GetCategoryCounts)
school_data.traveled.categorical.sum <- data.frame(catVars,school_data.traveled.categorical.sum)
names(school_data.traveled.categorical.sum)[2]<-paste("counts_of_categories")
ggplot(data = school_data.traveled.categorical.sum, aes(x = counts_of_categories)) + scale_x_continuous(name = "Percent Values NA") + scale_y_continuous(name="Number of Predictor Vars") +  geom_bar(binwidth = 0.05, fill = "firebrick4", color = "dodgerblue2") + ggtitle("Distribution of Missing Values on All Predictors") + theme(plot.title = element_text(lineheight=.8, face="bold"))

ImputeByZeroList <-school_data.traveled.numeric.sum$Variable[school_data.traveled.numeric.sum$MissPerc>0.70]
ImputeByMeanList <-school_data.traveled.numeric.sum$Variable[school_data.traveled.numeric.sum$MissPerc<=0.70]

#replace NAs with means
continuous_impute_by_mean <- data.frame(sapply(school_data.traveled[ImputeByMeanList],ImputeByMean))

#replace NAs with zeroes
continuous_impute_by_zero <-school_data.traveled[ImputeByZeroList]
continuous_impute_by_zero[is.na(continuous_impute_by_zero)]<-0

#put it all back together
school_data.traveled.xform <-data.frame(continuous_impute_by_mean,continuous_impute_by_zero,school_data.traveled[catVars])



#SPOILER ALERT:
#The following predictors have a singular value (most likely missing), and cause the glm function to fail
#HasBilingualEducation,HasMediaServices,HasTechnologyServices
#remove them here to continue on with 'full' model
#Plus just remove the things that aren't going to be used for prediction regardless
#Organization_id,PID,OrganizationName,City,ZipCode
#and things we really wouldn't 'know' for predicting
#ACPPaxInLastTwoYears,UniqueACGLLastTwoYears,Score
school_data.traveled.naive<-subset(school_data.traveled.xform, select=-c(HasBilingualEducation,HasMediaServices,HasTechnologyServices,Organization_id,PID,OrganizationName,City,ZipCode,ACPPaxInLastTwoYears,UniqueACGLLastTwoYears,Score,Prev6YrPPax,UniqueACGLPrevSixYears))

#you need to change the base to represent the starting model depending on forward or backward (or stepwise) selection
base<-glm(TraveledInLastTwoYears~Enrollment,data=school_data.traveled.naive)
model.min=formula(glm(TraveledInLastTwoYears~Enrollment,data=school_data.traveled.naive))
model.max=formula(glm(TraveledInLastTwoYears~.,data=school_data.traveled.naive))
school_travel.lm.varselection<-step(base,direction="forward",scope=list(lower=model.min, upper=model.max),trace=1)

save(base,model.min,model.max, file="SchoolProbModel.RData")

load(file="C:\\RStuff\\SchoolProbModel.RData")
#This is what we end up with
#forward selection:
#AIC=16618.97
#TraveledInLastTwoYears ~ Enrollment + GradeRange + WhiteStudents + 
#  HasAdvancedPlacement + DesignatedMarketArea + PercentOfStudentReceivingFreeReducedLunch + 
#  Grade10Students + CodedEnrollmentName + AlternativeSchoolCode + 
#  StudentsEnrolledInFreeReducedLunch + HasLibraryMediaCenter + 
#  InternationalBaccalaureateSchool + LifestyleIndicator + LowestGrade + 
#  SchoolCloseDate + BlackStudents + MagnetSchoolCode + HasAPSocialStudies + 
#  MetroName + AsianStudents + Grade9Students + Grade11Students + 
#  Grade12Students + HasAPCompSci + HasAPScience + HasSubstanceAbuseProgram + 
#  PovertyLevelCode + HasAdultEducation + HispanicStudents + 
#  HasTechnicalEducation + GiftedAndTalentedSchool + HighestGrade + 
#  IsCharterSchool + SpecialEducationStudents + State + OpenDateName + 
#  HasAPFineArts + Grade5Students + Grade7Students + TotalComputerCount + 
#  StudentComputerRatio + EnglishAsSecondLanguageSchool + PercentHispanic + 
#  IsYearRoundSchool + PovertyLevel + ClassroomTeachers + UngradedStudents




#remove the variables we don't want
#school_data.traveled.final<-subset(school_data.traveled.xform, select=-c(LanguagesSpoken,UngradedStudents,PreKStudents,KindergartenStudents,
#                                                                         Grade1Students,Grade2Students,Grade3Students,Grade4Students,
#                                                                         Grade5Students,Grade6Students,Grade7Students,Grade8Students,
#                                                                         Organization_id,PID,OrganizationName,ZipCode,SchoolOpenDate,
#                                                                         SchoolCloseDate,DesignatedMarketArea,CoreBasedStatisticalArea,
#                                                                         City,State,ACPPaxInLastTwoYears,UniqueACGLLastTwoYears,
#                                                                         City,ACPPaxInLastTwoYears,UniqueACGLLastTwoYears,
#                                                                         HasBilingualEducation,HasMediaServices,HasTechnologyServices,
#                                                                         CloseDateName))

school_data.traveled.final<-subset(school_data.traveled.xform, select=c(Organization_id,Score,TraveledInLastTwoYears,Enrollment,GradeRange,WhiteStudents,
                                                                          HasAdvancedPlacement,DesignatedMarketArea,PercentOfStudentReceivingFreeReducedLunch,
                                                                          Grade10Students,CodedEnrollmentName,AlternativeSchoolCode,
                                                                          StudentsEnrolledInFreeReducedLunch,HasLibraryMediaCenter,
                                                                          InternationalBaccalaureateSchool,LifestyleIndicator,LowestGrade,
                                                                          SchoolCloseDate,BlackStudents,MagnetSchoolCode,HasAPSocialStudies,
                                                                          MetroName,AsianStudents,Grade9Students,Grade11Students,
                                                                          Grade12Students,HasAPCompSci,HasAPScience,HasSubstanceAbuseProgram,
                                                                          PovertyLevelCode,HasAdultEducation,HispanicStudents,
                                                                          HasTechnicalEducation,GiftedAndTalentedSchool,HighestGrade,
                                                                          IsCharterSchool,SpecialEducationStudents,State,OpenDateName,
                                                                          HasAPFineArts,Grade5Students,Grade7Students,TotalComputerCount,
                                                                          StudentComputerRatio,EnglishAsSecondLanguageSchool,PercentHispanic,
                                                                          IsYearRoundSchool,PovertyLevel,ClassroomTeachers,UngradedStudents,first3zip))




#put the response and old score first - this is just for ease of use with the h20 stuff
col_idx<- grep("Score", names(school_data.traveled.final))
school_data.traveled.final <- school_data.traveled.final[, c(col_idx, (1:ncol(school_data.traveled.final))[-col_idx])]
col_idx<- grep("TraveledInLastTwoYears", names(school_data.traveled.final))
school_data.traveled.final <- school_data.traveled.final[, c(col_idx, (1:ncol(school_data.traveled.final))[-col_idx])]
#for the categoricals with high dimensionality, use the vtreat package to handle NA's, too many dimension, etc.
# (it's actually pretty cool - you should check it out - it transforms them into a continous value equal to their impact
# on the response, and creates a corresponding dummy for missing values)
myTreatments<-designTreatmentsC(school_data.traveled.final,c('State','DesignatedMarketArea','first3zip','LifestyleIndicator','GradeRange','SchoolCloseDate','PovertyLevelCode','LowestGrade'),'TraveledInLastTwoYears',1)
school_data.traveled.final.treat<-prepare(myTreatments,school_data.traveled.final)
school_data.traveled.final <-data.frame(school_data.traveled.final,school_data.traveled.final.treat[1:17])
#now that we have the transformed values, we can remove the original
school_data.traveled.final<-subset(school_data.traveled.final, select=-c(State,DesignatedMarketArea,first3zip,LifestyleIndicator,GradeRange,SchoolCloseDate,PovertyLevelCode,LowestGrade))


#split into test and train groups
indexes=sample(1:nrow(school_data.traveled.final),size=0.2*nrow(school_data.traveled.final))
test=school_data.traveled.final[indexes,]
train=school_data.traveled.final[-indexes,]

#myTreatments<-designTreatmentsC(train,c('State','DesignatedMarketArea','first3zip'),'TraveledInLastTwoYears',1)
#train.treat<-prepare(myTreatments,train)
#test.treat<-prepare(myTreatments,test)
#train <-data.frame(train,train.treat[1:5])
#test <-data.frame(test,test.treat[1:5])

#train<-subset(train, select=-c(State,DesignatedMarketArea,first3zip))
#test<-subset(test, select=-c(State,DesignatedMarketArea,first3zip))                                                                         
##################################################################
#Logistic Regression
##################################################################


school_travel.lm<-glm(TraveledInLastTwoYears ~ Enrollment + GradeRange_catN + GradeRange_catB + WhiteStudents + 
                        HasAdvancedPlacement + DesignatedMarketArea_catN + DesignatedMarketArea_catB + PercentOfStudentReceivingFreeReducedLunch +                         
                        Grade10Students + AlternativeSchoolCode + 
                        StudentsEnrolledInFreeReducedLunch + HasLibraryMediaCenter + 
                        InternationalBaccalaureateSchool + LifestyleIndicator_catN + LifestyleIndicator_catB + LowestGrade_catN + LowestGrade_catB + 
                        SchoolCloseDate_catN + SchoolCloseDate_catB + BlackStudents + MagnetSchoolCode + HasAPSocialStudies + 
                        MetroName + AsianStudents + Grade9Students + Grade11Students + 
                        Grade12Students + HasAPCompSci + HasAPScience + HasSubstanceAbuseProgram + 
                        PovertyLevelCode_lev_x. + PovertyLevelCode_lev_x.B + PovertyLevelCode_catN + PovertyLevelCode_catB + HispanicStudents + GiftedAndTalentedSchool +  
                        IsCharterSchool + SpecialEducationStudents + State_catN + OpenDateName +   
                        HasAPFineArts + Grade5Students + Grade7Students + TotalComputerCount + 
                        StudentComputerRatio + EnglishAsSecondLanguageSchool + PercentHispanic + 
                        PovertyLevel + ClassroomTeachers + UngradedStudents + first3zip_catN + first3zip_catB,data=train,family=binomial)


school_travel.lm.prob <- predict(school_travel.lm,test,type="response")
school_travel.lm.response = predict(school_travel.lm,type="response",newdata=test)
school_travel.lm.pred <- prediction(as.numeric(school_travel.lm.response)-1, test$TraveledInLastTwoYears)
school_travel.lm.perf <- performance(school_travel.lm.pred, "tpr", "fpr" )
school_travel.lm.auc <- performance(school_travel.lm.pred,"auc")
#set the threshold depending on tolerance for false pos vs false neg
mat=confusion.matrix(test$TraveledInLastTwoYears,school_travel.lm.response,threshold=0.5)
mat
omission(mat)
sensitivity(mat)
specificity(mat)
prop.correct(mat)





##################################################################
#Naive Bayes
##################################################################

#running model

#first draft
#school_travel.nb <- naiveBayes(TraveledInLastTwoYears ~ UniqueACGLPrevSixYears + Prev6YrPPax + 
#                                 Enrollment + NumberOfMacComputers + NumberOfPCs + PercentWhite + 
#                                 PercentBlack + PercentHispanic + Grade10Students + Grade12Students + 
#                                 AsianStudents + SpecialEducationPercentage + TotalComputerCount + 
#                                 PercentOfStudentReceivingFreeReducedLunch + MetroName + CodedEnrollmentName + 
#                                 OrgSubType + IsCharterSchool + MagnetSchoolCode + HasAdvancedPlacement + 
#                                 HasAPMath + HasAPSocialStudies + HasAPCompSci + HasSubstanceAbuseProgram + 
#                                 YoYDistrictEnrollmentChange + HasLibraryMediaCenter + StudentComputerRatio + State, data=train)

school_travel.nb <- naiveBayes(TraveledInLastTwoYears ~ Enrollment + GradeRange_catN + GradeRange_catB + WhiteStudents + 
                                 HasAdvancedPlacement + DesignatedMarketArea_catN + DesignatedMarketArea_catB + PercentOfStudentReceivingFreeReducedLunch +                         
                                 Grade10Students + AlternativeSchoolCode + 
                                 StudentsEnrolledInFreeReducedLunch + HasLibraryMediaCenter + 
                                 InternationalBaccalaureateSchool + LifestyleIndicator_catN + LifestyleIndicator_catB + LowestGrade_catN + LowestGrade_catB + 
                                 SchoolCloseDate_catN + SchoolCloseDate_catB + BlackStudents + MagnetSchoolCode + HasAPSocialStudies + 
                                 MetroName + AsianStudents + Grade9Students + Grade11Students + 
                                 Grade12Students + HasAPCompSci + HasAPScience + HasSubstanceAbuseProgram + 
                                 PovertyLevelCode_lev_x. + PovertyLevelCode_lev_x.B + PovertyLevelCode_catN + PovertyLevelCode_catB + HispanicStudents + GiftedAndTalentedSchool +  
                                 IsCharterSchool + SpecialEducationStudents + State_catN + OpenDateName +   
                                 HasAPFineArts + Grade5Students + Grade7Students + TotalComputerCount + 
                                 StudentComputerRatio + EnglishAsSecondLanguageSchool + PercentHispanic + 
                                 PovertyLevel + ClassroomTeachers + UngradedStudents + first3zip_catN + first3zip_catB, data=train)


#predict the test set with the trained model
school_travel.nb.prob <- predict(school_travel.nb,test,type="raw")
school_travel.nb.response = predict(school_travel.nb,type="class",newdata=test)
school_travel.nb.pred <- prediction(as.numeric(school_travel.nb.response)-1, test$TraveledInLastTwoYears)
school_travel.nb.perf <- performance(school_travel.nb.pred, "tpr", "fpr" )
school_travel.nb.auc <- performance(school_travel.nb.pred,"auc")
#set the threshold depending on tolerance for false pos vs false neg
mat=confusion.matrix(test$TraveledInLastTwoYears,school_travel.nb.prob[,2],threshold=0.50)
mat
omission(mat)
sensitivity(mat)
specificity(mat)
prop.correct(mat)


##################################################################
#SVM
##################################################################

#running model

school_travel.svm<-svm(TraveledInLastTwoYears ~ Enrollment + GradeRange_catN + GradeRange_catB + WhiteStudents + 
                         HasAdvancedPlacement + DesignatedMarketArea_catN + DesignatedMarketArea_catB + PercentOfStudentReceivingFreeReducedLunch +                         
                         Grade10Students + AlternativeSchoolCode + 
                         StudentsEnrolledInFreeReducedLunch + HasLibraryMediaCenter + 
                         InternationalBaccalaureateSchool + LifestyleIndicator_catN + LifestyleIndicator_catB + LowestGrade_catN + LowestGrade_catB + 
                         SchoolCloseDate_catN + SchoolCloseDate_catB + BlackStudents + MagnetSchoolCode + HasAPSocialStudies + 
                         MetroName + AsianStudents + Grade9Students + Grade11Students + 
                         Grade12Students + HasAPCompSci + HasAPScience + HasSubstanceAbuseProgram + 
                         PovertyLevelCode_lev_x. + PovertyLevelCode_lev_x.B + PovertyLevelCode_catN + PovertyLevelCode_catB + HispanicStudents + GiftedAndTalentedSchool +  
                         IsCharterSchool + SpecialEducationStudents + State_catN + OpenDateName +   
                         HasAPFineArts + Grade5Students + Grade7Students + TotalComputerCount + 
                         StudentComputerRatio + EnglishAsSecondLanguageSchool + PercentHispanic + 
                         PovertyLevel + ClassroomTeachers + UngradedStudents + first3zip_catN + first3zip_catB,data=train,kernel="radial",probability=TRUE)


school_travel.svm.prob <- predict(school_travel.svm,test, probability = TRUE)
school_travel.svm.response = predict(school_travel.svm,type="response",newdata=test)
school_travel.svm.pred <- prediction(as.numeric(school_travel.svm.response), test$TraveledInLastTwoYears)
school_travel.svm.perf <- performance(school_travel.svm.pred, "tpr", "fpr" )
school_travel.svm.auc <- performance(school_travel.svm.pred,"auc")
#set the threshold depending on tolerance for false pos vs false neg
mat=confusion.matrix(test$TraveledInLastTwoYears,school_travel.svm.response,threshold=0.50)
mat
omission(mat)
sensitivity(mat)
specificity(mat)
prop.correct(mat)


##################################################################
#ADA
##################################################################

#running model

school_travel.ada<-ada(TraveledInLastTwoYears ~ Enrollment + GradeRange_catN + GradeRange_catB + WhiteStudents + 
                         HasAdvancedPlacement + DesignatedMarketArea_catN + DesignatedMarketArea_catB + PercentOfStudentReceivingFreeReducedLunch +                         
                         Grade10Students + AlternativeSchoolCode + 
                         StudentsEnrolledInFreeReducedLunch + HasLibraryMediaCenter + 
                         InternationalBaccalaureateSchool + LifestyleIndicator_catN + LifestyleIndicator_catB + LowestGrade_catN + LowestGrade_catB + 
                         SchoolCloseDate_catN + SchoolCloseDate_catB + BlackStudents + MagnetSchoolCode + HasAPSocialStudies + 
                         MetroName + AsianStudents + Grade9Students + Grade11Students + 
                         Grade12Students + HasAPCompSci + HasAPScience + HasSubstanceAbuseProgram + 
                         PovertyLevelCode_lev_x. + PovertyLevelCode_lev_x.B + PovertyLevelCode_catN + PovertyLevelCode_catB + HispanicStudents + GiftedAndTalentedSchool +  
                         IsCharterSchool + SpecialEducationStudents + State_catN + OpenDateName +   
                         HasAPFineArts + Grade5Students + Grade7Students + TotalComputerCount + 
                         StudentComputerRatio + EnglishAsSecondLanguageSchool + PercentHispanic + 
                         PovertyLevel + ClassroomTeachers + UngradedStudents + first3zip_catN + first3zip_catB,data=train)

school_travel.ada.prob <- predict(school_travel.ada,test,type="prob")
school_travel.ada.response = predict(school_travel.ada,type="vector",newdata=test)
school_travel.ada.pred <- prediction(as.numeric(school_travel.ada.response)-1, test$TraveledInLastTwoYears)
school_travel.ada.perf <- performance(school_travel.ada.pred, "tpr", "fpr" )
school_travel.ada.auc <- performance(school_travel.ada.pred,"auc")
#set the threshold depending on tolerance for false pos vs false neg
mat=confusion.matrix(test$TraveledInLastTwoYears,school_travel.ada.prob[,2],threshold=0.55)
mat
omission(mat)
sensitivity(mat)
specificity(mat)
prop.correct(mat)

##################################################################
#GBM
##################################################################

#running model


school_travel.gbm <- gbm(as.vector(TraveledInLastTwoYears) ~ Enrollment + GradeRange_catN + GradeRange_catB + WhiteStudents + 
                                     HasAdvancedPlacement + DesignatedMarketArea_catN + DesignatedMarketArea_catB + PercentOfStudentReceivingFreeReducedLunch +                         
                                     Grade10Students + AlternativeSchoolCode + 
                                     StudentsEnrolledInFreeReducedLunch + HasLibraryMediaCenter + 
                                     InternationalBaccalaureateSchool + LifestyleIndicator_catN + LifestyleIndicator_catB + LowestGrade_catN + LowestGrade_catB + 
                                     SchoolCloseDate_catN + SchoolCloseDate_catB + BlackStudents + MagnetSchoolCode + HasAPSocialStudies + 
                                     MetroName + AsianStudents + Grade9Students + Grade11Students + 
                                     Grade12Students + HasAPCompSci + HasAPScience + HasSubstanceAbuseProgram + 
                                     PovertyLevelCode_lev_x. + PovertyLevelCode_lev_x.B + PovertyLevelCode_catN + PovertyLevelCode_catB + HispanicStudents + GiftedAndTalentedSchool +  
                                     IsCharterSchool + SpecialEducationStudents + State_catN + OpenDateName +   
                                     HasAPFineArts + Grade5Students + Grade7Students + TotalComputerCount + 
                                     StudentComputerRatio + EnglishAsSecondLanguageSchool + PercentHispanic + 
                                     PovertyLevel + ClassroomTeachers + UngradedStudents + first3zip_catN + first3zip_catB,data=train,n.trees = 1000,shrinkage = 0.001,bag.fraction = 0.5,train.fraction = 1.0,cv.folds=5)
#optimal number of trees
gbm.perf(school_travel.gbm)

school_travel.gbm.prob <- predict(school_travel.gbm,test,n.trees=500)
school_travel.gbm.response = predict(school_travel.gbm,type="response",newdata=test,n.trees=500)
school_travel.gbm.pred <- prediction(school_travel.gbm.response, test$TraveledInLastTwoYears)
school_travel.gbm.perf <- performance(school_travel.gbm.pred, "tpr", "fpr" )
school_travel.gbm.auc <- performance(school_travel.gbm.pred,"auc")
#set the threshold depending on tolerance for false pos vs false neg
mat=confusion.matrix(test$TraveledInLastTwoYears,school_travel.gbm.prob,threshold=0.55)
mat
omission(mat)
sensitivity(mat)
specificity(mat)
prop.correct(mat)




##################################################################
#RF
##################################################################


school_travel.rf <- randomForest(TraveledInLastTwoYears ~ Enrollment + GradeRange_catN + GradeRange_catB + WhiteStudents + 
                                   HasAdvancedPlacement + DesignatedMarketArea_catN + DesignatedMarketArea_catB + PercentOfStudentReceivingFreeReducedLunch +                         
                                   Grade10Students + AlternativeSchoolCode + 
                                   StudentsEnrolledInFreeReducedLunch + HasLibraryMediaCenter + 
                                   InternationalBaccalaureateSchool + LifestyleIndicator_catN + LifestyleIndicator_catB + LowestGrade_catN + LowestGrade_catB + 
                                   SchoolCloseDate_catN + SchoolCloseDate_catB + BlackStudents + MagnetSchoolCode + HasAPSocialStudies + 
                                   MetroName + AsianStudents + Grade9Students + Grade11Students + 
                                   Grade12Students + HasAPCompSci + HasAPScience + HasSubstanceAbuseProgram + 
                                   PovertyLevelCode_lev_x. + PovertyLevelCode_lev_x.B + PovertyLevelCode_catN + PovertyLevelCode_catB + HispanicStudents + GiftedAndTalentedSchool +  
                                   IsCharterSchool + SpecialEducationStudents + State_catN + OpenDateName +   
                                   HasAPFineArts + Grade5Students + Grade7Students + TotalComputerCount + 
                                   StudentComputerRatio + EnglishAsSecondLanguageSchool + PercentHispanic + 
                                   PovertyLevel + ClassroomTeachers + UngradedStudents + first3zip_catN + first3zip_catB,
                                   data=train, ntree=100, nodesize=5)

#predict the test set with the trained model
school_travel.rf.prob <- predict(school_travel.rf,test,type="prob")
school_travel.rf.response = predict(school_travel.rf,type="response",newdata=test)
school_travel.rf.pred <- prediction(as.numeric(school_travel.rf.response), test$TraveledInLastTwoYears)
school_travel.rf.perf <- performance(school_travel.rf.pred, "tpr", "fpr" )
school_travel.rf.auc <- performance(school_travel.rf.pred,"auc")
#set the threshold depending on tolerance for false pos vs false neg
mat=confusion.matrix(test$TraveledInLastTwoYears,school_travel.rf.prob[,2],threshold=0.55)
mat
omission(mat)
sensitivity(mat)
specificity(mat)
prop.correct(mat)

school_travel.rf.prob.2015 <- predict(school_travel.rf,gotime,type="prob")
#school_travel.rf.response.2015 = predict(school_travel.rf,type="response",newdata=gotime)



##################################################################
#h2o -deep learning neural net
##################################################################


## Start a local cluster with 7 cores and 12GB RAM
localH2O = h2o.init(ip = "localhost", port = 54321, startH2O = TRUE, max_mem_size = '12g', nthreads = 7)

## Convert data into H2O
#tnTrain and tnTest are my own personal data frames
h2o_train <- as.h2o(localH2O, train)
h2o_test <- as.h2o(localH2O, test)



#travel probability model
h2oDNN <- h2o.deeplearning(x = 4:60, # column numbers for predictors
                           y = 1, # column number for label
                           data = h2o_train, # data in H2O format
                           classification = TRUE,
                           autoencoder = FALSE,
                           #                           nfolds=10,
                           activation = "RectifierWithDropout", #There are several options here
                           input_dropout_ratio = 0.1, # % of inputs dropout
                           hidden_dropout_ratios = c(0.5,0.5,0.5,0.5,0.5), # % for nodes dropout
                           l2=.0005, #l2 penalty for regularization
                           seed=5,
                           hidden = c(100,100,200,100,100), # four layers of 100 nodes
                           variable_importances=TRUE,
                           epochs = 10) # max. no. of epochs


## Using the DNN model for predictions
school_travel.dnn <- h2o.predict(h2oDNN, h2o_test)
## Converting H2O format into data frame
school_travel.dnn.df <- as.data.frame(school_travel.dnn)

school_travel.dnn.pred <- prediction(school_travel.dnn.df$predict, test$TraveledInLastTwoYears)
school_travel.dnn.perf <- performance(school_travel.dnn.pred, "tpr", "fpr" )
school_travel.dnn.auc <- performance(school_travel.dnn.pred,"auc")

mat=confusion.matrix(test$TraveledInLastTwoYears,school_travel.dnn.df$X1,threshold=0.5)
mat
omission(mat)
sensitivity(mat)
specificity(mat)
prop.correct(mat)


#write out the predicted probabilities to a file

school.probs<- data.frame(test[,1:3])

school.probs<-cbind(school.probs,data.frame(school_travel.lm.prob))
names(school.probs)[4]<-paste("LR")

school.probs<-cbind(school.probs,data.frame(school_travel.nb.prob[,2]))
names(school.probs)[5]<-paste("NB")

school.probs<-cbind(school.probs,data.frame(school_travel.ada.prob[,2]))
names(school.probs)[6]<-paste("ada")

school.probs<-cbind(school.probs,data.frame(school_travel.rf.prob[,2]))
names(school.probs)[7]<-paste("rf")

school.probs<-cbind(school.probs,data.frame(attr(school_travel.svm.prob,"probabilities")[,2]))
names(school.probs)[8]<-paste("svm")

school.probs<-cbind(school.probs,data.frame(school_travel.gbm.response))
names(school.probs)[9]<-paste("gbm")

school.probs<-cbind(school.probs,school_travel.dnn.df$X1)
names(school.probs)[10]<-paste("DLNN")

write.csv(school.probs, file = "School_Probabilities_all.csv")
