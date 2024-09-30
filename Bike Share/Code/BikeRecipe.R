library(tidymodels)
library(vroom)
library(lubridate)

biketrain <- vroom("train.csv")
biketest <- vroom("test.csv")

# Clean data
biketrain <- subset(biketrain, select = c(-casual,
                                          -registered)) |> # Remove casual and registered variables
  mutate(count = log(count)) # Change count to log(count)

# Create recipe
bike_recipe <- recipe(count ~ ., data = biketrain) |> # Set model formula and dataset
  step_mutate(weather = ifelse(weather == 4, 3, weather)) |> # Recode weather category 4 to 3
  step_mutate(weather = factor(weather, # Make weather a factor
                               levels = c('1', '2', '3'), # Set levels
                               labels = c("Clear", "Cloudy", "Rainy"))) |> # Rename levels
  step_time(datetime, features = c("hour")) |> # Extract hour from datetime
  step_mutate(datetime_hour = factor(datetime_hour)) |>
  step_mutate(season = factor(season, # Make season a factor
                              levels = c('1', '2', '3', '4'), # Set levels
                              labels = c("Spring", "Summer", "Fall", "Winter"))) |> # Rename levels
  step_rm(windspeed) # Remove windspeed variable

bike_prep <- prep(bike_recipe) # Sets up preprocessing 
bake(bike_prep, new_data = biketrain) 

# Define linear model
bike_lm <- linear_reg() |> 
  set_engine("lm") |> 
  set_mode("regression")

# Combine into a workflow and fit
bike_workflow <- workflow() |> 
  add_recipe(bike_recipe) |> 
  add_model(bike_lm) |> 
  fit(data = biketrain)

# Run all steps on test data
bike_preds <- predict(bike_workflow, new_data = biketest)

# Prep for Kaggle submission
kaggle_sub <- bike_preds %>% 
  bind_cols(., biketest) |> # Bind predictions with test data
  select(datetime, .pred) |> # Keep datetime and prediction variables
  rename(count = .pred) |> # Rename pred to count for submission to Kaggle
  mutate(count = exp(count)) |> # Back-transform count
  mutate (count = pmax(0, count)) |> # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime)))

# Write out the file
vroom_write(x = kaggle_sub, file = "./RecipePreds.csv", delim = ",")
