


library(pscl)

zinb <- read.csv("https://stats.idre.ucla.edu/stat/data/fish.csv")
zinb <- within(zinb, {
  nofish <- factor(nofish)
  livebait <- factor(livebait)
  camper <- factor(camper)
})


m1 = zeroinfl(count ~ child + camper | persons, data = zinb)

print(m1)

summary(m1)

vcov(m1)
