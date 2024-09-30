library(tidymodels)
library(vroom)

# Read in data
biketrain <- vroom("train.csv")
biketest <- vroom("test.csv")

# Clean data
biketrain <- subset(biketrain, select = c(-casual,
                                          -registered)) |> # Remove casual and registered variables
  mutate(count = log(count)) # Change count to log(count)

# Create a recipe
bike_recipe <- recipe(count ~ ., data = biketrain) |> 
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |> # Recode weather category 4 to 3
  step_mutate(weather = factor(weather, # Make weather a factor
                               levels = 1:3, # Set levels
                               labels = c("Clear", "Cloudy", "Rainy"))) |> # Rename levels
  step_time(datetime, features = c("hour")) |> # Extract hour from datetime
  step_mutate(datetime_hour = factor(datetime_hour)) |>
  step_rm(datetime) |> 
  step_mutate(season = factor(season, # Make season a factor
                              levels = 1:4, # Set levels
                              labels = c("Spring", "Summer", "Fall", "Winter"))) |> # Rename levels
  step_rm(windspeed) # Remove windspeed variable

# Create random forest model
bike_forest <- rand_forest(mtry = tune(),
                           min_n=tune(),
                           trees=500) |> #Type of model
  set_engine("ranger") |> # What R function to use
  set_mode("regression")

# Create workflow
forest_workflow <- workflow() |> 
  add_recipe(bike_recipe) |> 
  add_model(bike_forest)

# Finalize parameters (use if not specifying mtry range)
# forest_param <- extract_parameter_set_dials(bike_forest) |> 
#  finalize(biketrain)

# Set up grid of tuning values
grid_tuning <- grid_regular(mtry(range=c(1,20)),
               min_n(),
               levels = 5)

# Set up K-fold cross-validation
folds <- vfold_cv(biketrain, v = 5, repeats = 1)

CV_results <- forest_workflow |> 
  tune_grid(resamples = folds,
            grid = grid_tuning,
            metrics = metric_set(rmse, mae, rsq))

# Find best tuning parameters
best_tuning <- CV_results |> 
  select_best(metric = "rmse")

# Finalize workflow and predict
final_workflow <- forest_workflow |> 
  finalize_workflow(best_tuning) |> 
  fit(data = biketrain)

forest_preds <- final_workflow |> 
  predict(new_data = biketest)

# Format for Kaggle submission
kaggle_sub <- forest_preds %>% 
  bind_cols(., biketest) |> # Bind predictions with test data
  select(datetime, .pred) |> # Keep datetime and prediction variables
  rename(count = .pred) |> # Rename pred to count for submission to Kaggle
  mutate(count = exp(count)) |> # Back-transform count
  mutate (count = pmax(0, count)) |> # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime)))

# Write out file
vroom_write(x = kaggle_sub, file = "./ForestPreds.csv", delim = ",")
