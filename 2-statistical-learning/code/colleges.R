# Read the colleges data in.
college = read.csv('college.csv')

# Label rows in the R dataframe using the first column, and remove that column
# from the matrix.
rownames(college) = college[,1]
college = college[,-1]
