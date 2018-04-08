

type_var = 'count'
preds0 = data.frame(yhat = predict(m1, type = type_var))
preds1 = data.frame(yhat = zip_pred(m1, type = type_var));
head(data.frame(preds0, preds1), 20) # output matches

d = DebTrivedi;
m = zeroinfl(ofp ~ hosp + health + numchron + gender + 
                school + privins | health, data = d)

m$count_X = model.matrix(m0$terms$count, m0$model, contrasts = m0$contrasts$count);
m$zero_X = model.matrix(m0$terms$zero, m0$model, contrasts = m0$contrasts$zero);
type_var = 'response'
preds0 = data.frame(yhat0 = predict(m0, type = type_var))
preds1 = data.frame(yhat1 = zip_pred(m0, type = type_var))
head(data.frame(preds0, preds1))

# zip: zeroinfl() model object
zip_pred = function(m = zip, newdata, type) {
  
  if (missing(newdata)) {
    rval = m$fitted.values;
    
    if (type != 'response') {
      X    = m$count_X;
      Z    = m$zero_X;
      # determine offset
      if (is.null(m$offset$count)) {
        offsetx = rep(0, nrow(X));
      } else {
        offsetx = m$offset$count
      }
      
      if (is.null(m$offset$zero)) {
        offsetz = rep(0, nrow(Z));
      } else {
        offsetz = m$offset$zero;
      }
      # finished determining offset
      
      mu  = exp(X %*% m$coefficients$count + offsetx)[1];
      phi = m$linkinv(Z %*% m$coefficients$zero + offsetz)[1]
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
  
  if (type == 'prob') {
    
    if (!is.null(m$y)) {
      y = m$y; 
    } else {
      print("Predicted Probabilities requires non-null values for y");
    }
    
    yUnique = 0:max(y);
    nUnique = length(yUnique);
    rval = matrix(NA, nrow = length(rval), ncol = nUnique);
    dimnames(rval) = list(rownames(X), yUnique);
    
    switch(m$dist,
           "poisson" = {
             rval[, 1] = phi + (1-phi) * exp(-mu)
             for(i in 2:nUnique) rval[,i] = (1-phi) * dpois(yUnique[i], lambda = mu)
           },
           "negbin" = {
             theta = object$theta
             rval[, 1] = phi + (1-phi) * dnbinom(0, mu = mu, size = theta)
             for(i in 2:nUnique) rval[,i] = (1-phi) * dnbinom(yUnique[i], mu = mu, size = theta)
           },
           "geometric" = {
             rval[, 1] = phi + (1-phi) * dnbinom(0, mu = mu, size = 1)
             for(i in 2:nUnique) rval[,i] = (1-phi) * dnbinom(yUnique[i], mu = mu, size = 1)
           })
  } # end if() -- 'prob' case
  
  return(rval)
} # end of zip_pred() function

