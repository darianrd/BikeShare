library(ggplot2)
library(patchwork)
library(vroom)
biketrain <- vroom("train.csv")

# Scatterplot of temperature vs. count
Var1 <- ggplot(data = biketrain, aes(x = temp,
                                     y = count)) +
  geom_point(color = "lightblue4") +
  labs(x = "Temperature in Celsius",
       y = "Number of rentals") +
  theme_minimal()

# Bar chart of season
Var2 <- ggplot(data = biketrain, aes(x = season)) +
  geom_bar(fill = "lightblue",
           color = "lightblue4") +
  labs(x = "Season",
       y = "Total number of riders") +
  ylim(0, 3500) +
  theme_minimal()

# Box plot of workdays vs. weekends
Var3 <- ggplot(data = biketrain, aes(group = workingday,
                                     x = workingday,
                                     y = registered)) +
  geom_boxplot(fill = "lightblue",
              color = "lightblue4") +
  labs(x = "Workday vs. Weekend",
       y = "Number of registered riders") +
  theme_minimal()

# Bar plot of weather
Var4 <- ggplot(data = biketrain, aes(x = weather)) +
  geom_bar(fill = "lightblue",
           color = "lightblue4") +
  labs(x = "Weather",
       y = "Total number of riders") +
  theme_minimal()

# Create four-panel ggplot
(Var1 + Var2)/(Var3 + Var4)