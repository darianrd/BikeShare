library(tidymodels)
library(vroom)
biketrain <- vroom("train.csv")
biketest <- vroom("test.csv")

# Set up and fit linear regression model
lm <- linear_reg() |> # Type of model
  set_engine("lm") |>  # Engine = What R function to use
  set_mode("regression") |> # Regression = Quantitative response
  fit(formula = log(count) ~ season + holiday + workingday + weather + atemp +
        humidity + windspeed,
      data = biketrain)

# Generate predictions using linear model
bike_predictions <- predict(lm,
                            new_data = biketest) # Use fit to predict
bike_predictions # Look at output

# Format predictions for submission to Kaggle
kaggle_sub <- bike_predictions %>% 
  bind_cols(., biketest) |> # Bind predictions with test data
  select(datetime, .pred) |> # Keep datetime and prediction variables
  rename(count = .pred) |> # Rename pred to count for submission to Kaggle
  mutate(count = exp(count)) |> # Back-transform count
  mutate (count = pmax(0, count)) |> # Pointwise max of (0, prediction)
  mutate(datetime = as.character(format(datetime))) # Needed for right format on Kaggle

# Write out the file
vroom_write(x = kaggle_sub, file = "./LinearPreds.csv", delim = ",")
