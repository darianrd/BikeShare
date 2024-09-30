library(tidymodels)
library(vroom)
library(poissonreg)

biketrain <- vroom("train.csv")
biketest <- vroom("test.csv")

bike_pois <- poisson_reg() %>% # Type of model
  set_engine("glm") %>% # GLM = Generalized linear model
  set_mode("regression") %>%
  fit(formula = count ~ season + holiday + workingday + weather + temp + atemp +
        humidity + windspeed, data = biketrain)

# Generate predictions using linear model
bike_preds <- predict(bike_pois,
                      new_data = biketest)
bike_preds

# Format predictions for submission to Kaggle
pois_kaggle_sub <- bike_preds %>%
  bind_cols(., biketest) %>% # Bind predictions with test data
  select(datetime, .pred) %>% # Just keep datetime and .pred variables
  rename(count = .pred) %>% # Rename .pred to count for submission to Kaggle
  mutate(datetime = as.character(format(datetime))) # Needed for right format

# Write out the file
vroom_write(x = pois_kaggle_sub,
            file = "./PoissonPreds.csv",
            delim = ",")