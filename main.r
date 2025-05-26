

#  Load Required Packages

packages <- c("ggplot2", "dplyr", "caret", "randomForest", "rpart", "rpart.plot", "pROC", "corrplot", "reshape2")
install_if_missing <- function(pkg) {
  if (!require(pkg, character.only = TRUE)) install.packages(pkg, dependencies = TRUE)
}
invisible(lapply(packages, install_if_missing))


# Load the Dataset

data <- read.csv("Dataset.csv")



# Initial Data Exploration

summary(data)
str(data)
cat("Total missing values:", sum(is.na(data)), "\n")


# Data Cleaning & Preparation

data$Churn <- as.factor(data$Churn)
data$International.plan <- as.factor(data$International.plan)
data$Voice.mail.plan <- as.factor(data$Voice.mail.plan)
data_clean <- data %>% select(-State, -Area.code)


# Exploratory Data Analysis (EDA)


## Churn Distribution
ggplot(data, aes(x = Churn)) + 
  geom_bar(fill = c("skyblue", "tomato")) +
  ggtitle("Churn Distribution") +
  theme_minimal()

## Numeric Variables Distribution
num_vars <- sapply(data_clean, is.numeric)
melted <- melt(data_clean[, num_vars])

ggplot(melted, aes(x = value)) + 
  facet_wrap(~ variable, scales = "free", ncol = 3) +
  geom_histogram(binwidth = NULL, fill = "#4682B4", color = "black", alpha = 0.8) +
  theme_minimal() +
  labs(title = "Histogram of Numeric Features")

## Correlation Matrix
cor_matrix <- cor(data_clean[, num_vars])
corrplot(cor_matrix, method = "color", type = "upper", tl.cex = 0.8)


# Split Dataset into Training & Testing Sets

set.seed(123)  # ensures reproducible sampling
split_index <- createDataPartition(data_clean$Churn, p = 0.7, list = FALSE)
train_data <- data_clean[split_index, ]
test_data <- data_clean[-split_index, ]

# Train Decision Tree Model

tree_model <- rpart(Churn ~ ., data = train_data, method = "class")
rpart.plot(tree_model, extra = 104, fallen.leaves = TRUE, main = "Decision Tree for Churn")

# Train a Random Forest Model

forest_model <- randomForest(Churn ~ ., data = train_data, ntree = 100, importance = TRUE)
print(forest_model)
varImpPlot(forest_model, main = "Top Predictors (Random Forest)")


# Predict Probabilities on Test Data

rf_probs <- predict(forest_model, test_data, type = "prob")[,2]
tree_probs <- predict(tree_model, test_data, type = "prob")[,2]

#  Evaluate Models Using ROC Curves

rf_roc <- roc(test_data$Churn, rf_probs)
tree_roc <- roc(test_data$Churn, tree_probs)

plot(rf_roc, col = "navy", lwd = 2, main = "Comparison of ROC Curves")
lines(tree_roc, col = "darkred", lwd = 2)
legend("bottomright", legend = c("Random Forest", "Decision Tree"), 
       col = c("navy", "darkred"), lwd = 2)

cat("Random Forest AUC:", auc(rf_roc), "\n")
cat("Decision Tree AUC:", auc(tree_roc), "\n")
