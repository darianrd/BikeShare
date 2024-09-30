library(tidymodels)
library(vroom)

biketrain <- vroom("train.csv")
biketest <- vroom("test.csv")

# Clean data
biketrain <- subset(biketrain, select = c(-casual,
                                          -registered)) |> # Remove casual and registered variables
  mutate(count = log(count)) # Change count to log(count)

# Create a recipe
bike_recipe_pen <- recipe(count ~ ., data = biketrain) |> 
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
  step_rm(windspeed) |> # Remove windspeed variable
  step_dummy(all_nominal_predictors()) |> # Make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean = 0, SD = 1

# Create penalized regression model
penreg_model <- linear_reg(penalty = tune(), mixture = tune()) |> # Set model and tuning
  set_engine("glmnet") # Fit elastic net regression

penreg_workflow <- workflow() |>  
  add_recipe(bike_recipe_pen) |>  
  add_model(penreg_model)

# Grid of values to tune over
grid_tuning <- grid_regular(penalty(),
                            mixture(),
                            levels = 5)

# Split data for cross-validation
folds <- vfold_cv(biketrain, v = 5, repeats = 1)

# Run cross-validation
CV_results <- penreg_workflow |> 
  tune_grid(resamples = folds,
            grid = grid_tuning,
            metrics = metric_set(rmse, mae, rsq))

# Plot results
collect_metrics(CV_results) |> # Gathers metrics into DF
  filter(.metric == "rmse") |> 
  ggplot(aes(x= penalty, y = mean, color = factor(mixture))) +
  geom_line()

# Find best tuning parameters
best_tuning <- CV_results |> 
  select_best(metric = "rmse")

# Finalize workflow and fit
final_workflow <- penreg_workflow |> 
  finalize_workflow(best_tuning) |> 
  fit(data = biketrain)

# Predict
CV_preds <- final_workflow |> 
  predict(new_data = biketest)

# Format for Kaggle submission
kaggle_sub <- CV_preds %>% 
  bind_cols(., biketest) |> # Bind predictions with test data
  select(datetime, .pred) |> # Keep datetime and prediction variables
  rename(count = .pred) |> # Rename pred to count for submission to Kaggle
  mutate(count = exp(count)) |> # Back-transform count
  mutate (count = pmax(0, count)) |> # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime)))

# Write out file
vroom_write(x = kaggle_sub, file = "./CVPreds.csv", delim = ",")
