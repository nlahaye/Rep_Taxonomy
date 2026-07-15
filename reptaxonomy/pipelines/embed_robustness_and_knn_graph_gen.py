import os as os
import pathlib
import pprint
import numpy as np
import pandas as pd

from reptaxonomy.intercompare.knn_graph_gen import build_knn_graph
from reptaxonomy.util.embed_scale_and_subset import get_indices, rescale_embed 
from reptaxonomy.robustness.cluster_analysis import train_and_gen_projection
from reptaxonomy.robustness.intrinsic_dimension import estimate_id


import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


#TODO - import silhouette computation? Currently within the pipeline
from sklearn import metrics



def main(cfg):

    # fix all random seeds
    fix_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"



    #TODO expected input format


    # get datasets
    raw_test_dataset: RawGeoFMDataset = instantiate(cfg.dataset, split="test")
    test_dataset = GeoFMDataset(raw_test_dataset, test_preprocessor)

    test_loader = DataLoader(
        test_dataset,
        # sampler=DistributedSampler(test_dataset),
        num_workers=cfg.test_num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    #TODO - end dataset/model initialiation - to change over to gfmtools style


    #Get/make directories
    embed_dir = os.path.join(cfg["embed_dir"],cfg["dataset_name"])
    indices_dir = os.path.join(cfg["out_dir"],cfg["dataset_name"])
    out_dir = os.path.join(indices_dir, cfg["encoder"])
    
    if not os.path.isdir(indices_dir):
        os.makedirs(indices_dir, exist_ok = True)

    if not os.path.isdir(out_dir):
        os.makedirs(out_dir, exist_ok = True)

    n_runs = 3
    silhouettes = {}
    all_rows_id = []
    for i in range(n_runs):
        #Structure embedding test data
        embed_full = None
        target_full = None    
  
        build_knn = True    
        #if i != 1:
        #    build_knn = False

        runs_subdir = "run_" + str(i) 

        for batch_idx, data in enumerate(test_loader):
            if "filename" in data:
                image, target, image_fname, meta = data["image"], data["target"],data["filename"], data['metadata']
                image_fname = image_fname[0]
            else:
                image, target, meta = data["image"], data["target"], data['metadata']
                image_fname = meta['image_filename'][0]

            #Load embeddings for single input file
            embed_fname = os.path.join(embed_dir, cfg["encoder"], "test", "embd_" + os.path.splitext(image_fname)[0] + ".npy")
            crop_info_fname = os.path.join(embed_dir, cfg["encoder"], "test", "crop_info_" + os.path.splitext(image_fname)[0] + ".npy")
            embed = np.load(embed_fname)

            crop_info = None
            if os.path.exists(crop_info_fname):
                crop_info = np.load(crop_info_fname, allow_pickle=True).item()

            #print(crop_info)

            #print(embed_fname,  cfg.dataset.img_size)
            for k, v in image.items():
                crop_info = crop_info[k]
                img_size = v[:, :, 0, :, :].shape
            img_size = img_size[-1] 
 
            #print(crop_info)
    
            #Rescale embedding to original dimension
            print("RESCALE PRE", embed.min(), embed.max())
            embed, target = rescale_embed(embed, img_size, device, target, crop_info)
            print("RESCALE POST", embed.min(), embed.max(), embed.shape)

            #Flatten dimensions, except feature/channel dim

            #target = target.flatten()

            #Get subset and apply indices, sampling each class available - subsetting done due to compuational complexity of tasks

            indices_subdir = os.path.join(indices_dir, runs_subdir)
            os.makedirs(indices_subdir, exist_ok=True)

            indices = get_indices(cfg, image_fname, target, indices_subdir, (crop_info[0][-2], crop_info[0][-1]))
        
       
            sub_embed = embed[indices,:]
            sub_target = target[indices]

            print("SUB SIZE", sub_embed.shape)
            #Merge individual subsets together
            if embed_full is None:
                embed_full = sub_embed
                target_full = sub_target
            else:
                embed_full = np.concatenate((sub_embed, embed_full))
                target_full = np.concatenate((sub_target, target_full))


        print("HERE", embed_full.min(), embed_full.max())
        out_subdir = os.path.join(out_dir, runs_subdir)
        os.makedirs(out_subdir, exist_ok=True) 


        methods = ["MLE", "TwoNN", "TLE", "DANCo", "FisherS", "MOM", "CorrInt", "ESS", "MiND_ML", "MiND_KL", "MADA"]
        k_values = [5,10,20]

        print("Estimating id")
        id_results = estimate_id(embed_full, methods, k_values)

        for row in id_results:

            row.update(
                 {
                     "model": cfg["encoder"],
                     "n_samples": int(embed_full.shape[0]),
                     "ambient_dim": int(embed_full.shape[1]),
                      #"feature_manifest": str(detail_path),
                 }
            )

            all_rows_id.append(row)

 
        for projection in ["tsne", "umap", "pca"]:
                #projection = "tsne" #"umap" #"pca"
                projection_data, reducer = train_and_gen_projection(embed_full, out_subdir, cfg, projection=projection)

                #del embed_full

                #Shift indices to start w/ zero - we can then use GeoTiff files for output / viz
                shift_1 = abs(min(projection_data[:,0]))
                shift_2 = abs(min(projection_data[:,1]))

                projection_data[:,0] = projection_data[:,0] + shift_1
                projection_data[:,1] = projection_data[:,1] + shift_2

                #Scale data to expand for viz.
                projection_data = (projection_data*10).astype(np.int32)

                max_ind_1 = int(max(projection_data[:,0]))
                max_ind_2 = int(max(projection_data[:,1]))
                final_projection = np.zeros((max_ind_1+1, max_ind_2+1), dtype=np.int32) - 1.0

                for i in range(target_full.shape[0]):
                    final_projection[int(projection_data[i,0]), int(projection_data[i,1])] = target_full[i]
 
                ras_meta = {'driver': 'GTiff', 'dtype': 'int32', 'nodata': -1, 'width': final_projection.shape[1], 'height': final_projection.shape[0], 'count': 1, 'tiled': False, 'interleave': 'band'}
          
                silhouette = metrics.silhouette_score(projection_data, target_full)

                if projection not in silhouettes:
                    silhouettes[projection] = { cfg["encoder"] : [silhouette] }
                elif cfg["encoder"] not in silhouettes[projection]:
                    silhouettes[projection][cfg["encoder"]] = [silhouette] 
                else:
                    silhouettes[projection][cfg["encoder"]].append(silhouette)

                print("SILHOUETTE:", projection, cfg["encoder"], silhouette)
                out_file = os.path.join(out_subdir, cfg["encoder"] + "." + projection.upper()  + "_Labels.tif")
                with rasterio.open(out_file, 'w', **ras_meta) as dst:
                    dst.write(final_projection, 1)
    
          
                knn_fpath = os.path.join(out_subdir, cfg["encoder"] + "." + cfg["dataset_name"] + "." + projection.upper() + ".knn_graph.zarr")
                if build_knn and not os.path.exists(knn_fpath): #TODO allow for overwrite flag
                    print("Building KNN Graph", knn_fpath)
                    build_knn_graph(projection_data, knn_fpath)

    out_sil = []
    for key in silhouettes:
        for key2 in silhouettes[key]:
            print("SILHOUETTE STATS", key, key2, np.mean(silhouettes[key][key2]), np.std(silhouettes[key][key2]))
            out_sil.append({"projection": key, "model":key2, "mean_sil": np.mean(silhouettes[key][key2]), "std_sil": np.std(silhouettes[key][key2])})
    df = pd.DataFrame(out_sil)
    df.to_csv(os.path.join(out_dir, "silhouette_stats." + cfg["encoder"] + "." + cfg["dataset_name"] + ".csv"), index=False) 

    df2 = pd.DataFrame(all_rows_id)
    df2.to_csv(os.path.join(out_dir, "intrinsic_dimension_summary." + cfg["encoder"] + "." + cfg["dataset_name"] + ".csv"), index=False)

   
if __name__ == "__main__":
    main()
