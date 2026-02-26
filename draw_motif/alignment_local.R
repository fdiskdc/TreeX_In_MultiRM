library(Biostrings)
library(msa)

args <- commandArgs(trailingOnly = TRUE)

if (length(args) >= 5) {
  data_type <- args[1]
  RM <- args[2]
  length <- args[3]
  w <- args[4]
  k <- args[5]
} else {
  stop("Usage: Rscript alignment_local.R <data_type> <RM> <length> <w> <k>")
}

file_name <- paste(data_type, RM, length, w, k, sep = '_')
file_name <- paste(file_name, 'csv', sep = '.')
short_seq <- read.csv(file_name, header = FALSE)
short_seq <- DNAStringSet(as.character(short_seq$V1))
align <- msa(short_seq, gapOpening = 50000)
align_chars <- as.character(align@unmasked)

out_path <- paste(data_type, RM, length, w, 'aligned', sep = '_')
out_path <- paste(out_path, 'csv', sep = '.')
write.table(x = align_chars, file = out_path, sep = '\n', row.names = FALSE, col.names = FALSE)

cat("Alignment completed. Output saved to:", out_path, "\n")
