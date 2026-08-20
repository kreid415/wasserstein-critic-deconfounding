suppressMessages({library(Seurat); library(Matrix)})
D <- Sys.getenv("SEURAT_DIR")
cnt <- readMM(file.path(D,"counts.mtx")); cells <- read.csv(file.path(D,"cells.csv")); genes <- read.csv(file.path(D,"genes.csv"))
rownames(cnt) <- genes$gene; colnames(cnt) <- paste0("c", cells$cell)
obj <- CreateSeuratObject(counts=cnt); obj$batch <- cells$batch
lst <- SplitObject(obj, split.by="batch")
lst <- lapply(lst, function(x){ x <- NormalizeData(x, verbose=FALSE); FindVariableFeatures(x, nfeatures=2000, verbose=FALSE) })
feats <- SelectIntegrationFeatures(lst, verbose=FALSE)
mind <- min(sapply(lst, ncol)); dd <- 1:min(30, mind-1)
anch <- FindIntegrationAnchors(lst, anchor.features=feats, reduction="cca", dims=dd, verbose=FALSE)
integ <- IntegrateData(anchorset=anch, dims=dd, verbose=FALSE)
DefaultAssay(integ) <- "integrated"; integ <- ScaleData(integ, verbose=FALSE); integ <- RunPCA(integ, npcs=30, verbose=FALSE)
emb <- Embeddings(integ,"pca")[paste0("c", cells$cell), ]
write.csv(emb, file.path(D,"seurat_emb.csv")); cat("emb", dim(emb)[1], "x", dim(emb)[2], "\n")
