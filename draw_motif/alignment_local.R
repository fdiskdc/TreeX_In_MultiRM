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

# Construct directory path based on w and k parameters
# The directory name format is: train_hm6A_w8_k5 (from wid8 -> w8, top5 -> k5)
dir_name <- paste(data_type, RM, paste0('w', sub('wid', '', w)), paste0('k', k), sep = '_')

# Get the script's directory (this works when script is sourced directly)
script_path <- normalizePath(sub('--file=', '', commandArgs(trailingOnly = FALSE)[grep('^--file=', commandArgs(trailingOnly = FALSE))]))
script_dir <- dirname(script_path)

# Base directory is draw_motif/draw_motif/dir_name relative to script location
base_dir <- file.path(script_dir, 'draw_motif', dir_name)

# Input file path
file_name <- paste(data_type, RM, length, w, k, sep = '_')
file_name <- paste(file_name, 'csv', sep = '.')
input_path <- file.path(base_dir, file_name)

# Check if input file exists
if (!file.exists(input_path)) {
  stop(paste("Input file not found:", input_path))
}

short_seq <- read.csv(input_path, header = FALSE)
short_seq <- DNAStringSet(as.character(short_seq$V1))
align <- msa(short_seq, gapOpening = 50000)
align_chars <- as.character(align@unmasked)

# Output file path
out_file <- paste(data_type, RM, length, w, 'aligned', sep = '_')
out_file <- paste(out_file, 'csv', sep = '.')
out_path <- file.path(base_dir, out_file)

write.table(x = align_chars, file = out_path, sep = '\n', row.names = FALSE, col.names = FALSE)

cat("Alignment completed. Output saved to:", out_path, "\n")
