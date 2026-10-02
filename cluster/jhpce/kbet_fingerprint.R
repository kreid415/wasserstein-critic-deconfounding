# Fingerprint of the installed R kBET package: md5 over the deparsed formals + body of every object in its
# namespace (same R version => identical text for identical code). Used to show that the kBET installed on
# JHPCE (theislab/kBET commit afc5f431) is the same code as the local Rlib_kbet copy.
# Usage: R_LIBS=<lib> Rscript cluster/jhpce/kbet_fingerprint.R
suppressPackageStartupMessages(library(kBET))
ns <- asNamespace("kBET")
nms <- sort(ls(ns, all.names = TRUE))
txt <- vapply(nms, function(n) {
  o <- get(n, envir = ns)
  if (is.function(o)) paste(c(n, deparse(formals(o)), deparse(body(o))), collapse = "\n") else paste(n, class(o)[1])
}, "")
cat(sprintf("kBET version=%s objects=%d md5=%s lib=%s R=%s\n", as.character(packageVersion("kBET")), length(nms),
            digest::digest(paste(txt, collapse = "\n"), algo = "md5", serialize = FALSE),
            dirname(find.package("kBET")), R.version.string))
