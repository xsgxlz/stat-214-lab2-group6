
# Description: A script for generating several EDA plots based on the 3 labeled
# MISR images after being processed by clean.py.


#>---------------------- Load Packages

library(tidyverse)   # ggplot2 for plotting and dplyr for data manipulation
library(GGally)      # ggpairs for scatterplot matrices
library(rlang)       # as_name function, used in scatterplot matrix helper function
library(glue)        # for string manipulation
library(ggtext)      # for markdown text rendering
library(patchwork)   # for combining subplots


#>---------------------- Data Cleaning Functions

# adds the file name column to the corresponding dataframe
clean <- function(df, name=NA) {
  df_clean <- df %>%
    mutate(label = factor(label),
           file = name)
  
  return(df_clean)
}


#>---------------------- Load and Process Data

# file names of the 3 labeled images
file1 <- "O012791.csv"
file2 <- "O013257.csv"
file3 <- "O013490.csv"

# image names of the 3 labeled images
image_names <- c("O012791", "O013257", "O013490")

# load each image's data
df1 <- read.csv(file1)
df2 <- read.csv(file2)
df3 <- read.csv(file3)

# add the file name to each dataframe
df1 <- clean(df1, image_names[1])
df2 <- clean(df2, image_names[2])
df3 <- clean(df3, image_names[3])

# combine into a single dataframe
df4 <- rbind(df1, df2, df3) %>%
  mutate(file = as.factor(file))


#>---------------------- Helper Functions

# set color of markdown text
fColor <- function(text, color) {glue("<span style='color:{color};'>{text}</span>")}

# set size of markdown text
fSize <- function(subtitle, size) {glue("<span style='font-size:{size}pt'>{subtitle}</span>")}

# create a markdown plot title
createTitle <- function(title, subtitle) {paste(title, "  \n", fSize(subtitle, 11), sep="")}

# randomly sample n indices such that the label proportions are balanced
samp <- function(n, label) {
  noclouds <- sample(which(label == -1), n/3)
  unlabeled <- sample(which(label == 0), n/3)
  clouds <- sample(which(label == 1), n/3)
  c(noclouds, unlabeled, clouds)
}


#>---------------------- Global Theme Options

# color defaults
RED <- "maroon3"
BLUE <- "skyblue2"
DARKGRAY <- "gray30"
LIGHTGRAY <- "gray95"


#>---------------------- Radiance Angle Scatterplot Matrices

# set title text
title1 <- glue("**Scatterplot Matrix of Radiance Angles (Image {image_names[1]})**")
title2 <- glue("**Scatterplot Matrix of Radiance Angles (Image {image_names[2]})**")
title3 <- glue("**Scatterplot Matrix of Radiance Angles (Image {image_names[3]})**")

# set subtitle text
subt <- paste(
  "Comparing equal subsamples of pixels labeled",
  fColor("*Clouds*", BLUE),
  "vs.",
  fColor("*Non-Clouds*", RED),
  "by experts"
)

# create markdown titles
corr1.title <- createTitle(title1, subt)
corr2.title <- createTitle(title2, subt)
corr3.title <- createTitle(title3, subt)


# set labeles for each column name
angle_labels <- c("raDF" = "DF",
                  "raCF" = "CF",
                  "raBF" = "BF",
                  "raAF" = "AF",
                  "raAN" = "AN")

# define content of each scatterplot
upperfun <- function(data, mapping) {
  # get variables
  x_var <- as_name(mapping$x)
  y_var <- as_name(mapping$y)
  c <- cor(data[[x_var]], data[[y_var]])
  
  # calculate the x and y positions for the text
  x_range <- range(data[[x_var]], na.rm = TRUE)
  y_range <- range(data[[y_var]], na.rm = TRUE)
  x_pos <- x_range[1] + 0.30 * diff(x_range)
  y_pos <- y_range[2] - 0.05 * diff(y_range)
  
  # return ggplot object
  ggplot(data = data, mapping = mapping) +
    geom_point(alpha = 0.2) +
    geom_text(aes(label = paste("r =",round(c,3)), x=x_pos, y=y_pos),
              color = DARKGRAY, size = 4) +
    scale_color_manual(values = c("-1" = RED, "0" = BLUE)) 
}

# define scatterplot matrix function
corrplot <- function(df, title) {
  ggpairs(df, aes(color=label, alpha=0.05),
          columns = 1:5,
          lower = "blank",
          upper = list(continuous = wrap(upperfun)),
          labeller = as_labeller(angle_labels)) +
  scale_color_manual(name = "Expert Labels",
                     labels = c("Not Cloud", "Cloud"),
                     values = c("-1" = RED, "1" = BLUE)) +
  scale_fill_manual(name = "Expert Labels",
                    labels = c("Not Cloud", "Cloud"),
                    values = c("-1" = RED, "1" = BLUE)) +
  labs(title=title) +
  theme_bw() +
  theme(plot.title = element_markdown(lineheight = 1.1, hjust=0.5),
        axis.text = element_blank(),
        axis.ticks = element_blank())
    
}

# create scatterplot matrices
corr1 <- df1 %>%
  slice(sample(samp(15000, label))) %>%
  filter(label != 0) %>%
  select(6:11) %>%
  corrplot(corr1.title)

corr2 <- df2 %>%
  slice(sample(samp(15000, label))) %>%
  filter(label != 0) %>%
  select(6:11) %>%
  corrplot(corr2.title)

corr3 <- df3 %>%
  slice(sample(samp(15000, label))) %>%
  filter(label != 0) %>%
  select(6:11) %>%
  corrplot(corr3.title)

# save scatterplot matrices as SVG files
ggsave("corr1.svg", corr1,  width = 8, height = 8)
ggsave("corr2.svg", corr2,  width = 8, height = 8)
ggsave("corr3.svg", corr3,  width = 8, height = 8)

# save scatterplot matrices as PNG files
ggsave("corr1.png", corr1,  width = 8, height = 8)
ggsave("corr2.png", corr2,  width = 8, height = 8)
ggsave("corr3.png", corr3,  width = 8, height = 8)


#>---------------------- Constructed Feature Scatterplot Matrices

# draw balanced samples of rows for each dataframe
i1 <- sample(samp(15000, df1$label))
i2 <- sample(samp(15000, df2$label))
i3 <- sample(samp(15000, df3$label))

# filter each dataframe, removing unlabeled points
df1_sub <- df1 %>% slice(i1) %>% filter(label != 0)
df2_sub <- df2 %>% slice(i2)%>% filter(label != 0)
df3_sub <- df3 %>% slice(i3) %>% filter(label != 0)

# create combined dataframe
df4_sub <- rbind(df1[i1, ], df2[i2, ], df3[i3, ]) %>%
  # filter relevant variables
  filter(label != 0) %>%
  mutate(file = as.factor(file)) %>%
  select(NDAI, SD, CORR, label, file) %>%
  
  # triplicate rows and add comparison variable
  mutate(id = row_number()) %>%
  uncount(3) %>%
  group_by(id) %>%
  mutate(comp = c("NDAI vs. SD", "SD vs. CORR", "NDAI vs. CORR")) %>%
  ungroup() %>%
  select(-id) %>%
  
  # set x=var1 and y=var2 for each comparison
  mutate(var1 = 0, var2 = 0,
         var1 = ifelse(comp == "NDAI vs. SD",   SD, var1),
         var1 = ifelse(comp == "SD vs. CORR",   CORR, var1),
         var1 = ifelse(comp == "NDAI vs. CORR", CORR, var1),
         var2 = ifelse(comp == "NDAI vs. SD",   NDAI, var2),
         var2 = ifelse(comp == "SD vs. CORR",   SD, var2),
         var2 = ifelse(comp == "NDAI vs. CORR", NDAI, var2)) %>%
  
  # add variables used to plot correlation coefficient labels
  group_by(file, comp) %>%
  mutate(c = cor(var1, var2),
         xmin = min(var1), xmax = max(var1),
         ymin = min(var2), ymax = max(var2),
         xrange = xmax - xmin,
         yrange = ymax - ymin) %>%
  ungroup()


# define function for creating each row of subplots
subplot <- function(data, title=NULL, subtitle=NULL) {
  
  ggplot(data, aes(x=`var1`, y=`var2`, color=`label`)) +
    geom_point(alpha=0.2, show.legend=FALSE) +
    geom_text(aes(label = paste("r =",round(c,3)),
                  x=xmax - 0.15 * xrange,
                  y=ymax - 0.05 * yrange),
              color = DARKGRAY, size = 4) +
    facet_wrap(.~comp, scales="free") +
    labs(title=title, subtitle=subtitle) +
    scale_color_manual(name="Expert Label",
                       values=c("-1"=RED, "1"=BLUE),
                       labels=c("Not Cloud", "Cloud")) +
    theme_bw() +
    theme(axis.text = element_blank(),
          axis.ticks= element_blank(),
          axis.title = element_blank(),
          plot.title = element_markdown(lineheight = 1.1, hjust=0.5),
          plot.subtitle = element_markdown(hjust=0.5))
}

# set titles and subtitles
title4 <- "**Scatterplot Matrix of NDAI, SD, & CORR**"
corr4.title <- createTitle(title4, subt)
corr41.subt <- glue("**Image {image_names[1]}**")
corr42.subt <- glue("**Image {image_names[2]}**")
corr43.subt <- glue("**Image {image_names[3]}**")

# create subplots
corr41 <- df4_sub %>% filter(file == image_names[1]) %>%
  subplot(corr4.title, corr41.subt)

corr42 <- df4_sub %>% filter(file == image_names[2]) %>%
  subplot(NULL, corr42.subt)

corr43 <- df4_sub %>% filter(file == image_names[3]) %>%
  subplot(NULL, corr43.subt)

# combine subplots
corr4 <- corr41 / corr42 / corr43

# save scatterplot matrix as SVG file
ggsave("corr4.svg", corr4,  width = 7, height = 7)

# save scatterplot matrix as PNG file
ggsave("corr4.png", corr4,  width = 7, height = 7)


#>---------------------- Map Plot

# set image labels
image_labels <- c("O012791" = "Image O012791",
                 "O013257" = "Image O013257",
                 "O013490" = "Image O013490")

# create map
map1 <- df4 %>%
  ggplot(aes(x=x, y=y, fill=label)) +
  geom_tile(color=NA) +
  facet_grid(cols = vars(file), labeller = as_labeller(image_labels)) +
  scale_fill_manual(name="Expert Labels",
                    labels = c("Not Cloud", "Unlabeled", "Cloud"),
                    values = c("-1" = RED, "0" = LIGHTGRAY, "1" = BLUE)) +
  coord_fixed() +
  labs(title="Map of Cloud Distribution According to Expert Labels",
       y="y-coordinate",
       x="x-coordinate") +
  theme_bw() +
  theme(plot.title = element_text(hjust = 0.5),
        panel.ontop = TRUE,
        panel.background = element_rect(color = NA, fill = NA),
        panel.grid.major = element_line(color=DARKGRAY, linewidth=0.1),
        panel.grid.minor = element_line(color=DARKGRAY, linewidth=0.1))

# save map as SVG file
ggsave("map.svg", map1,  width = 9, height = 4)

# save map as PNG file
ggsave("map.png", map1,  width = 9, height = 4)
