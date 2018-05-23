library(readr)
library(cluster)
library(dbscan)
ai2013papers <- read_csv("ai2013_papers.csv")
ai2013papers.reduced <- ai2013_papers[, 2:12]


### ========== kMeans ==========
set.seed(20)
ai2013.kmeans.fit <- kmeans(ai2013papers.reduced, 8)

kmeansPlot <- clusplot(ai2013papers.reduced, ai2013.kmeans.fit$cluster, main = '2D representation of ai2013.kmeans.fit',
         color = TRUE, shade = TRUE,
         labels = 2, lines = 0)

# confusion matrix
table(ai2013papers$type, ai2013.kmeans.fit$cluster)


### ========== Hierarchical clustering ==========

d <- dist(ai2013papers.reduced, method = "euclidean")
H.fit <- hclust(d, method = "ward.D")

# display dendogram
plot(H.fit) 
groups <- cutree(H.fit, k = 8) 
rect.hclust(H.fit, k = 8, border = "red") 

# confusion matrix
table(ai2013papers$type, groups)


### ========== DBSCAN ==========
# find knee point
kNNPlot <- kNNdistplot(as.matrix(ai2013papers.reduced), k = 8)
abline(h = 90, col = "red")
kNNPlot

set.seed(1234)
ai2013.db.fit = dbscan(as.matrix(ai2013papers.reduced), 90, minPts = 2)
hullplot(as.matrix(ai2013papers.reduced), ai2013.db.fit$cluster)

# confusion matrix
table(ai2013papers$type, ai2013.db.fit$cluster)