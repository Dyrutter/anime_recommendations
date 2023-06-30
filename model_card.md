The primary approach for this model is collaborative filtering, a sort of matrix factorization, which generates recommendations based on the similarity of a user’s anime ratings as opposed to the similarity of anime to other anime or users to other users. A utility matrix is used to assess that relationship, with each individual user a row, each anime a column and a body consisting of scaled ratings. Recommendations are formulated by forecasting the ratings, or, filling in the blanks as seen in Figure 1.  














Figure 1: A utility matrix of user IDs (y axis) and their ratings of anime IDs (x axis). The blank spaces represent anime a user hasn’t rated.

The matrix is quite sparse, as nearly 18,000 anime exist, and few users have rated more than a couple hundred. In order to compensate for this, a new, complete rating matrix is surmised as the product of two smaller matrices. That is the rating matrix (C), which has number of user ID rows (n) and number of anime ID columns (m), is decomposed into two matrices representing user IDs and anime IDs respectively. The first matrix, (U), has (n) rows and (d) columns, where (d), known as an embedding vector, represents some set of features users might find important. The second matrix, (V), contains (d) rows and (m) columns, where (d) represents features of users that might lead them to enjoy certain animes. The embedding vectors can have any length so long as the length is consistent between (U) and (V). Through experimentation, I found 128 to be the optimal length. The dot product of these embedding matrices, UVT, creates a new matrix (R). The weights in each embedding vector are initialized randomly and adjusted via gradient descent until R best approximates M.

The proximity of R to M is assessed using root-mean-squared-error (RMSE): 
Sum over all known ratings (non-zero elements of M) the square of the difference between those ratings and the corresponding elements of R 
Compute the mean of those squares, and 
Compute the square root of the mean.

The smaller the error between the known entries in M and their respective values in R, the better R is assumed to approximate the unknown entries in M.

Neural networks work by minimizing estimation error with a loss algorithm, in this case gradient descent. This inherent optimization capability can be used to perform matrix factorization by projecting items into a latent space. The embedding layers map high-dimensional sparse sets of discrete features to dense arrays of real numbers in continuous spaces. In the case of my model, they project (n) users and (m) anime into (d)-dimensional vectors. Once the model is trained, ratings are predicted by the dot product between embeddings followed by running through a single Dense layer.  
