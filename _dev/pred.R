

# response: count
# count features: child, camper
# zero features:  persons
m1 = zeroinfl(count ~ child + camper | persons, data = zinb)

m1$fitted.values
m1$model


preds0 = data.frame(yhat = predict(m1, type = 'response'))
head(preds, 20)

preds1 = data.frame(yhat = zip_pred(m1, type = 'response'));
head(preds1, 20)

head(data.frame(preds, preds1), 20) # output matches

# count training matrix with intercept in first column
testX = model.matrix(m1$terms$count, m1$model, contrasts = m1$contrasts$count)
# zero training matrix with intercept in first column
testZ = model.matrix(m1$terms$zero,  m1$model, contrasts = m1$contrasts$zero)  

head(testX)
head(testZ)

m1$count_X = testX
m1$zero_X  = testZ

# zip: zeroinfl() model object
zip_pred = function(m = zip, newdata, type) {
  
  if (missing(newdata)) {
    rval = m$fitted.values;
    
    if (type != 'response') {
      
      rval = m$fitted.values;
      
      X = m$count_X;
      Z = m$count_Z;
      
      # determine offset
      if (is.null(m$offset$count)) {
        offsetx = rep(0, nrow(m$count_X));
      } else {
        offsetx = m$offset$count
      }
      
      if (is.null(m$offset$zero)) {
        offsetz = rep(0, nrow(m$zero_X));
      } else {
        offsetz = m$offset$zero;
      }
      # finished determining offset
      
      mu  = exp(X %*% m$coefficients$count + offsetx)[, 1];
      phi = m$linkinv(Z %*% m$coefficients$zero + offsetz)[, 1]
    } # no action if type == 'response' 
  } else { # corresponds to outer if
    print("non-missing newdata option not yet supported");
  } # end of outer if/else
  
  # predicted means for count/zero component
  if (type == 'count') {
    rval = mu;
  }
  if (type == 'zero') {
    rval = phi;
  }
  
  # if (type == 'prob')
  
  
  return(rval)
}

