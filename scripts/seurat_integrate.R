# Seurat CCA integration -> 30-dim integrated embedding, written to CSV.
suppressMessages({library(Seurat); library(Matrix)})
setwd(Sys.getenv("WCD_WD"))
cnt <- readMM("results/seurat_in/counts.mtx")            # genes x cells
cells <- read.csv("results/seurat_in/cells.csv"); genes <- read.csv("results/seurat_in/genes.csv")
rownames(cnt) <- genes$gene; colnames(cnt) <- paste0("c", cells$cell)
obj <- CreateSeuratObject(counts = cnt)
obj$batch <- cells$batch
lst <- SplitObject(obj, split.by = "batch")
lst <- lapply(lst, function(x){ x <- NormalizeData(x, verbose=FALSE);
        FindVariableFeatures(x, nfeatures=2000, verbose=FALSE) })
feats <- SelectIntegrationFeatures(lst, verbose=FALSE)
# CCA anchors; dims guarded to smallest batch
kf <- max(30, 200); mind <- min(sapply(lst, ncol))
anch <- FindIntegrationAnchors(lst, anchor.features=feats, reduction="cca",
         dims=1:min(30, mind-1), verbose=FALSE)
integ <- IntegrateData(anchorset=anch, dims=1:min(30, mind-1), verbose=FALSE)
DefaultAssay(integ) <- "integrated"
integ <- ScaleData(integ, verbose=FALSE); integ <- RunPCA(integ, npcs=30, verbose=FALSE)
emb <- Embeddings(integ, "pca")[paste0("c", cells$cell), ]   # reorder to original
write.csv(emb, "results/seurat_in/seurat_emb.csv")
cat("Seurat embedding:", dim(emb)[1], "x", dim(emb)[2], "\n")
