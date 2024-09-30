library(tidymodels)
library(vroom)
library(stacks)

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
  step_rm(windspeed) |> # Remove windspeed variable
  step_dummy(all_nominal_predictors()) |> # Make dummy variables
  step_normalize(all_numeric_predictors()) # Make mean = 0, SD = 1

# Set up K-fold cross-validation
folds <- vfold_cv(biketrain, v = 5, repeats = 1)

# Create a control grid
untuned_model <- control_stack_grid()
tuned_model <- control_stack_resamples()

# Create penalized regression model
penreg <- linear_reg(penalty = tune(),
                     mixture = tune()) |>
  set_engine("glmnet") 

# Create penalized regression workflow
penreg_workflow <- workflow() |> 
  add_recipe(bike_recipe) |> 
  add_model(penreg)

# Create tuning grid for penalized regression
penreg_tuning_grid <- grid_regular(penalty(),
                                   mixture(),
                                   levels = 5)

# Run cross-validation for penalized regression
penreg_model <- penreg_workflow |> 
  tune_grid(resamples = folds,
            grid = penreg_tuning_grid,
            metrics = metric_set(rmse, mae),
            control = untuned_model)

# Create linear regression model
linreg <- linear_reg() |> 
  set_engine("lm")

# Create linear regression workflow
linereg_workflow <- workflow() |> 
  add_model(linreg) |> 
  add_recipe(bike_recipe)

# Create resampling objects
linreg_model <- fit_resamples(linereg_workflow,
                              resamples = folds,
                              metrics = metric_set(rmse, mae, rsq),
                              control = tuned_model)

# Create random forest model
forest <- rand_forest(mtry = tune(),
                      min_n=tune(),
                      trees=500) |> 
  set_engine("ranger") |> 
  set_mode("regression")

# Create random forest workflow
forest_workflow <- workflow() |> 
  add_recipe(bike_recipe) |> 
  add_model(forest)

# Set up tuning grid for random forest
forest_tuning_grid <- grid_regular(mtry(range=c(1,30)),
                            min_n(),
                            levels = 5)

# Run cross-validation for random forest
forest_model <- forest_workflow |> 
  tune_grid(resamples = folds,
            grid = forest_tuning_grid,
            metrics = metric_set(rmse, mae, rsq),
            control = untuned_model)

# Include models to stack
bike_stack <- stacks() |> 
  add_candidates(penreg_model) |> 
  add_candidates(linreg_model) |> 
  add_candidates(forest_model)

# Fit stacked model
stack_models <- bike_stack |> 
  blend_predictions() |> # LASSO penalized regression meta-learner
  fit_members() # Fit the members to the dataset

# Use stacked data to get new prediction
stack_preds <- stack_models |>
  predict(new_data = biketest)

# Format for Kaggle submission
kaggle_sub <- stack_preds %>% 
  bind_cols(., biketest) |> # Bind predictions with test data
  select(datetime, .pred) |> # Keep datetime and prediction variables
  rename(count = .pred) |> # Rename pred to count for submission to Kaggle
  mutate(count = exp(count)) |> # Back-transform count
  mutate (count = pmax(0, count)) |> # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime)))

# Write out file
vroom_write(x = kaggle_sub, file = "./StackingPreds.csv", delim = ",")

