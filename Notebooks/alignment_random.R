library(Biostrings)
library(msa)

RM ='hm6A'

data_type <- '../Seqs/test'
length <- '51'
w <- 'wid8'
k <- 'top2'
file_name <- paste(data_type,RM,length,w,k,sep = '_')
file_name <- paste(file_name,'csv',sep = '.')

# 读取序列
short_seq <- read.csv(file_name, header = FALSE)

# ================= 核心修改 =================
# 随机抽取 5000 条序列以防止内存爆炸
set.seed(42) # 保证每次抽样结果一致
sample_size <- min(5000, nrow(short_seq)) 
short_seq <- short_seq[sample(nrow(short_seq), sample_size), , drop=FALSE]
# ============================================

short_seq <- DNAStringSet(as.character(short_seq$V1))

# 执行多序列比对
align <- msa(short_seq, gapOpening = 50000)
align_chars <- as.character(align@unmasked)

out_path <- paste(data_type,RM,length,w,'aligned',sep = '_')
out_path <- paste(out_path,'csv',sep = '.')
write.table(x = align_chars, file = out_path, sep = '\n', row.names = FALSE, col.names = FALSE)