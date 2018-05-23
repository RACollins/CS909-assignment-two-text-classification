# Create Iris data set with only versicolor and setosa species
# and convert Species column to Class with values 1 and -1
irisBin <- iris[iris$Species == 'versicolor' | iris$Species == 'setosa',]

# Label setosa as 1 and versicolor as -1
# Seperate features from class labels
speciesVector <- ifelse(irisBin$Species == "setosa", 1, -1)
featureVectors <- irisBin[, 1:4]

# Distance from hyperplane 
distanceFromPlane = function(x, w, b){
  return(sum(x*w) + b)
}

# Classify which side the instance lies
signClassify = function(x, w, b){
  d <- apply(x, 1, distanceFromPlane, w, b)
  sign <- ifelse(d > 0, 1, -1)
  return(sign)
}

# calculate euclidean normailisation 
eNorm = function(x){sqrt(sum(x^2))}

# Perceptron function takes a set of feature vectors and class labels
perceptron = function(x, Y, eta = 100){
  # initialise weigth vector w, intercept b, and number of updates k
  w = vector(length = ncol(x))
  b = 0
  R = max(apply(x, 1, eNorm))
  whileCondition = TRUE
  while(whileCondition){
    whileCondition = FALSE
    classifiedY <- signClassify(x, w, b)
    for (i in 1:nrow(x)){
      if (Y[i] != classifiedY[i]){
        w <- w + eta * Y[i] * x[i,]
        b <- b + eta * Y[i] * R^2
        whileCondition = TRUE
      }
    }
  }
  return(list(w=w/eNorm(w), b=b/eNorm(b)))
}

perceptron(featureVectors, speciesVector)