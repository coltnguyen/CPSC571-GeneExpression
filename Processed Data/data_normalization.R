# load libraries
library(DESeq2)

# set working directory
setwd('C:/Users/toque/Desktop/CPSC_571-Design_Implementation_Database_Systems/03-Assignments/Project/yy-Data')

# load count data
count_data <- read.csv('init_data.csv', header=TRUE, row.names=1)
head(count_data)

# load sample info
sample_info <- read.csv('init_data_design.csv', header=TRUE, row.names=1)

# make sure row names in sample_info match column names in count_data
all(colnames(count_data) %in% rownames(sample_info))

# are columns and rows in the same order?
all(colnames(count_data) == rownames(sample_info))

# create DESeq object, importing count data & sample info into it
dds <- DESeqDataSetFromMatrix(countData=count_data, colData=sample_info, design=~cancerType)

# display DESeq object
dds

# remove rows with low gene counts
# see why I picked '((997 / 8) * 10)' at these links:
# https://bioconductor.org/packages/release/bioc/vignettes/DESeq2/inst/doc/DESeq2.html#pre-filtering
# https://www.youtube.com/watch?v=OzNzO8qwwp0&list=PLJefJsd1yfhYa97VUXkQ4T90YH6voYjCn (9:02)
# https://www.youtube.com/watch?v=wPzeea1Do18&t=551s (15:28)
keep <- rowSums(counts(dds)) >= ((997 / 6) * 10)
dds <- dds[keep,]
dds

# perform normalization on the values
ddsnormalization <- DESeq(dds)

# write normalized values to file
normCounts <- counts(ddsnormalization, normalized=TRUE)
write.csv(normCounts, 'normalized_data.csv')
