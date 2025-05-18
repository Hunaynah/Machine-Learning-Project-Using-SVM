# ==========================================
# TELECOM CUSTOMER CHURN — EDA & SVM MODELING
# ==========================================

# 1. Load Libraries
library(tidyverse)
library(skimr)
library(ggplot2)
library(dplyr)
library(ggcorrplot)
library(caret)
library(scales)
library(caTools)
library(e1071)
library(ROCR)

# 2. Load the Dataset
df <- read.csv("telecom_customer_churn.csv",
               na.strings = c("", "NA", "N/A", "null", "Missing"))

# 3. Structure & Initial Overview
str(df)
summary(df)
cat("Rows:", nrow(df), " Columns:", ncol(df), "\n")

# 4. Missing Values Check
missing <- colSums(is.na(df))
missing[missing > 0]

# 5. Duplicates
duplicates <- sum(duplicated(df$Customer.ID))
cat("Duplicate Customer IDs:", duplicates, "\n")

# 6. Handle Missing Values
df$Avg.Monthly.Long.Distance.Charges[is.na(df$Avg.Monthly.Long.Distance.Charges) & df$Phone.Service == "No"] <- 0
df$Multiple.Lines[is.na(df$Multiple.Lines) & df$Phone.Service == "No"] <- "No service"
df$Avg.Monthly.GB.Download[is.na(df$Avg.Monthly.GB.Download) & df$Internet.Service == "No"] <- 0

internet_cols <- c("Internet.Type", "Online.Security", "Online.Backup", 
                   "Device.Protection.Plan", "Premium.Tech.Support", 
                   "Streaming.TV", "Streaming.Movies", "Streaming.Music", 
                   "Unlimited.Data")

for (col in internet_cols) {
  df[[col]][is.na(df[[col]]) & df$Internet.Service == "No"] <- "No service"
}

df$Churn.Reason   <- as.factor(ifelse(is.na(df$Churn.Reason),   "No Churn", df$Churn.Reason))
df$Churn.Category <- as.factor(ifelse(is.na(df$Churn.Category), "No Churn", df$Churn.Category))

# 7. Remove Unnecessary Columns
df <- df %>% select(-Customer.ID, -Latitude, -Longitude, -Zip.Code, -City)

# 8. Descriptive Statistics
numeric_vars <- df %>% select(where(is.numeric))
summary(numeric_vars)

# 9. Outlier Detection & Removal for Total.Revenue
Q1 <- quantile(df$Total.Revenue, 0.25)
Q3 <- quantile(df$Total.Revenue, 0.75)
IQR <- Q3 - Q1
lower_bound <- Q1 - 1.5 * IQR
upper_bound <- Q3 + 1.5 * IQR

df_no_outliers <- df %>%
  filter(Total.Revenue >= lower_bound & Total.Revenue <= upper_bound)

# 10. Churn Status Column for Visualization
df_no_outliers$Churn_Status <- factor(
  ifelse(df_no_outliers$Churn.Category == "No Churn", "No Churn", "Churn"),
  levels = c("No Churn", "Churn")
)

# 11. Visualization
# Pie Chart of Churn Distribution
ggplot(df_no_outliers, aes(x = "", fill = Churn_Status)) +
  geom_bar(width = 1) +
  coord_polar(theta = "y") +
  geom_text(
    stat = "count",
    aes(label = scales::percent(..count.. / sum(..count..))),
    position = position_stack(vjust = 0.5)
  ) +
  scale_fill_manual(values = c("lightgreen", "pink")) +
  labs(title = "Churn vs No Churn Distribution", fill = "") +
  theme_void()

# Histogram of Tenure by Churn
ggplot(df_no_outliers, aes(x = Tenure.in.Months, fill = Churn_Status)) +
  geom_histogram(binwidth = 6, position = "dodge") +
  labs(title = "Churn by Tenure", x = "Tenure (Months)", y = "Count")

# 12. Check SVM Assumptions
# A. Class Balance
table(df_no_outliers$Churn_Status)
prop.table(table(df_no_outliers$Churn_Status))

# B. Correlation
num_data <- df_no_outliers %>% select(where(is.numeric))
corr_matrix <- cor(num_data, use = "complete.obs")
ggcorrplot(corr_matrix, hc.order = TRUE, type = "lower", lab = TRUE)
corr_matrix

# Optional: Remove highly correlated variables
high_corr <- findCorrelation(corr_matrix, cutoff = 0.9)
if (length(high_corr) > 0) {
  num_data <- num_data[, -high_corr]
}

# C. Recheck missing
colSums(is.na(df_no_outliers))

# 13. Train/Test Split
set.seed(123)
split <- sample.split(df_no_outliers$Churn_Status, SplitRatio = 0.75)
training_set <- subset(df_no_outliers, split == TRUE)
test_set     <- subset(df_no_outliers, split == FALSE)

# 14. Feature Scaling
num_cols <- names(training_set)[sapply(training_set, is.numeric)]
pp <- preProcess(training_set[, num_cols], method = c("center", "scale"))
training_set[, num_cols] <- predict(pp, training_set[, num_cols])
test_set[, num_cols]     <- predict(pp, test_set[, num_cols])

# 15. SVM Model (Linear Kernel)
classifier <- svm(Churn_Status ~ 
                    Gender + Age + Married + Number.of.Dependents + Number.of.Referrals +
                    Tenure.in.Months + Avg.Monthly.Long.Distance.Charges + Avg.Monthly.GB.Download +
                    Phone.Service + Multiple.Lines + Internet.Service + Internet.Type +
                    Online.Security + Online.Backup + Device.Protection.Plan + Premium.Tech.Support +
                    Unlimited.Data + Contract + Paperless.Billing + Payment.Method +
                    Monthly.Charge + Total.Extra.Data.Charges + Total.Long.Distance.Charges,
                  data = training_set,
                  type = 'C-classification',
                  kernel = 'linear',
                  class.weights = c('Churn' = 4, 'No Churn' = 1.5))

# 16. Model Evaluation
predictions <- predict(classifier, newdata = test_set)
test_set$Churn_Status <- factor(test_set$Churn_Status, levels = c("No Churn", "Churn"))
predictions <- factor(predictions, levels = levels(test_set$Churn_Status))

confusionMatrix(predictions, test_set$Churn_Status)

# 17. ROC Curve & AUC
prob_predictions <- predict(classifier, test_set, decision.values = TRUE)
pred <- prediction(as.numeric(prob_predictions), as.numeric(test_set$Churn_Status))
perf <- performance(pred, "tpr", "fpr")

plot(perf, col = "blue", main = "ROC Curve")
auc <- performance(pred, measure = "auc")
auc_value <- auc@y.values[[1]]
cat("AUC:", auc_value, "\n")
legend("bottomright", legend = paste("AUC =", round(auc_value, 3)), col = "blue", lwd = 2)







# SVM Model with RBF Kernel
classifier_rbf <- svm(Churn_Status ~ 
                        Gender + Age + Married + Number.of.Dependents + Number.of.Referrals +
                        Tenure.in.Months + Avg.Monthly.Long.Distance.Charges + Avg.Monthly.GB.Download +
                        Phone.Service + Multiple.Lines + Internet.Service + Internet.Type +
                        Online.Security + Online.Backup + Device.Protection.Plan + Premium.Tech.Support +
                        Unlimited.Data + Contract + Paperless.Billing + Payment.Method +
                        Monthly.Charge + Total.Extra.Data.Charges + Total.Long.Distance.Charges+ Total.Revenue,
                      data = training_set,
                      type = 'C-classification',
                      kernel = 'radial',  # Using RBF Kernel
                      gamma = 0.1, cost = 1,
                      class.weights = c('Churn' = 4, 'No Churn' = 1.5))


# Model Evaluation
predictions_rbf <- predict(classifier_rbf, newdata = test_set)
test_set$Churn_Status <- factor(test_set$Churn_Status, levels = c("No Churn", "Churn"))
predictions_rbf <- factor(predictions_rbf, levels = levels(test_set$Churn_Status))

# Confusion Matrix for the RBF SVM
confusionMatrix(predictions_rbf, test_set$Churn_Status)

# ROC Curve & AUC for the RBF SVM
prob_predictions_rbf <- predict(classifier_rbf, test_set, decision.values = TRUE)
pred_rbf <- prediction(as.numeric(prob_predictions_rbf), as.numeric(test_set$Churn_Status))
perf_rbf <- performance(pred_rbf, "tpr", "fpr")

# Plotting the ROC Curve
plot(perf_rbf, col = "red", main = "ROC Curve for RBF Kernel")
auc_rbf <- performance(pred_rbf, measure = "auc")
auc_value_rbf <- auc_rbf@y.values[[1]]
cat("AUC for RBF Kernel:", auc_value_rbf, "\n")
legend("bottomright", legend = paste("AUC =", round(auc_value_rbf, 3)), col = "red", lwd = 2)







# 1. SVM Model with Polynomial Kernel
classifier_poly <- svm(Churn_Status ~ 
                         Gender + Age + Married + Number.of.Dependents + Number.of.Referrals +
                         Tenure.in.Months + Avg.Monthly.Long.Distance.Charges + Avg.Monthly.GB.Download +
                         Phone.Service + Multiple.Lines + Internet.Service + Internet.Type +
                         Online.Security + Online.Backup + Device.Protection.Plan + Premium.Tech.Support +
                         Unlimited.Data + Contract + Paperless.Billing + Payment.Method +
                         Monthly.Charge + Total.Extra.Data.Charges + Total.Long.Distance.Charges + Total.Revenue,
                       data = training_set,
                       type = 'C-classification',
                       kernel = 'polynomial',  # Use Polynomial Kernel
                       degree = 3,             # Polynomial degree (default is 3)
                       coef0 = 1,              # Scale parameter (typically a value between 0 and 1)
                       class.weights = c('Churn' = 4, 'No Churn' = 1.5))  # Class weights

# 2. Model Evaluation
predictions_poly <- predict(classifier_poly, newdata = test_set)
test_set$Churn_Status <- factor(test_set$Churn_Status, levels = c("No Churn", "Churn"))
predictions_poly <- factor(predictions_poly, levels = levels(test_set$Churn_Status))

# Confusion Matrix
conf_matrix_poly <- confusionMatrix(predictions_poly, test_set$Churn_Status)
print(conf_matrix_poly)

# 3. ROC Curve & AUC for Polynomial Kernel
prob_predictions_poly <- predict(classifier_poly, test_set, decision.values = TRUE)
pred_poly <- prediction(as.numeric(prob_predictions_poly), as.numeric(test_set$Churn_Status))
perf_poly <- performance(pred_poly, "tpr", "fpr")

# Plotting the ROC Curve for Polynomial Kernel
plot(perf_poly, col = "green", main = "ROC Curve for Polynomial Kernel")

# AUC Calculation for Polynomial Kernel
auc_poly <- performance(pred_poly, measure = "auc")
auc_value_poly <- auc_poly@y.values[[1]]
cat("AUC for Polynomial Kernel:", auc_value_poly, "\n")

