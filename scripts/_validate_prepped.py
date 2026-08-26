import scanpy as sc, glob, os
for f in sorted(glob.glob('results/scvi_single/*_prepped.h5ad')):
    ds = os.path.basename(f).replace('_prepped.h5ad', '')
    a = sc.read_h5ad(f)
    ok = ('counts' in a.layers) and ('batch_key' in a.uns) and ('celltype_key' in a.uns)
    print('VAL', ds, 'n_obs', a.n_obs, 'nb', a.obs['batch'].nunique(), 'counts', 'counts' in a.layers, 'uns_ok', ok)
