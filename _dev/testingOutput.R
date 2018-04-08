


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

preds = data.frame(yhat = predict(m1, type = 'response'))
head(preds)

preds1 = data.frame(yhat = predict(m1, type = 'zero'))
head(preds1)

# response: count
# count features: child, camper
# zero features:  persons
m1 = zeroinfl(count ~ child + camper | persons, data = zinb)

m1$fitted.values
m1$model

# count training matrix with intercept in first column
testX = model.matrix(m1$terms$count, m1$model, contrasts = m1$contrasts$count)
# zero training matrix with intercept in first column
testZ = model.matrix(m1$terms$zero,  m1$model, contrasts = m1$contrasts$zero)  

m1$count_X = testX
m1$zero_X  = testZ






