import NLPEngine as engine
Data="Summarization systems often have additional evidence they can utilize in order to specify the most important topics of document(s). " \
     "For example, when summarizing blogs, there are discussions or comments coming after the blog post that are good sources of information to determine which parts of the blog are critical and interesting. " \
     " In scientific paper summarization, there is a considerable amount of information such as cited papers and conference information which can be leveraged to identify important sentences in the original paper. " \
     "Although we don’t have to calculate the Eigenvalues and Eigenvectors by hand but it is important to understand the inner workings to be able to confidently use the algorithms. " \
     "Furthermore, It is very straightforward to calculate eigenvalues and eigenvectors in Python. " \
     "Once we have calculated eigenvalues, we can calculate the Eigenvectors of matrix A by using Gaussian Elimination. " \
     "Gaussian elimination is about converting the matrix to row echelon form. " \
     "Finally, it is about solving the linear system by back substitution. " \
     "The above equation states that we need to find eigenvalue (lambda) and eigenvector (x) such that when we multiply a scalar lambda (eigenvalue) to the vector x (eigenvector) then it should equal to the linear transformation of the matrix A once it is scaled by vector x (eigenvector)."

c=engine.NlpEngineStart(Data)
summaryData=c.summary()
print(summaryData)