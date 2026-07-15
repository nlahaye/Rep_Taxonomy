
import numpy as np

import torch
import torch.nn.functional as F

import math
import zarr

#Get per-scene indices for stratified (by label counts) subsetting of embeddings
def get_indices(cfg, image_fname, target, indices_dir, shape_info) -> None:
 
    #Get or set indices used to subset embeddings from individual files
    index_fname = os.path.join(indices_dir, os.path.splitext(image_fname)[0] + ".indices.zarr")

    final_inds = None

    #TODO - Get from gfmtools?
    n_classes = cfg.dataset.num_classes
    ignore_class = cfg.dataset.ignore_index
    #END TODO

    if not os.path.exists(index_fname):
        sub_inds = []

        for i in range(0, n_classes+1):

            if i == ignore_class:
                continue

            sub_inds_init = np.where(target == i)[0]
            sub_inds = []

            #only use middle of scene to account for cropping
            #print(target.shape)

            miny = int(0.15 * shape_info[0])
            maxy = int(0.85 * shape_info[0])
            minx = int(0.15 * shape_info[1])
            maxx = int(0.85 * shape_info[1])
            #print(sub_inds_init, (int(miny) * shape_info[1]), (int(maxy) * shape_info[1]))
            for si in range(len(sub_inds_init)):
                if sub_inds_init[si] > (int(miny) * shape_info[1]) and \
                    sub_inds_init[si] < (int(maxy) * shape_info[1]) and \
                    sub_inds_init[si] % shape_info[0] > minx and \
                    sub_inds_init[si] % shape_info[0] < maxx:

                        sub_inds.append(sub_inds_init[si])

            if len(sub_inds) < 1:
                continue


            #Ensure each class is represented - can add stratification later
            if len(sub_inds)  > cfg["label_file_subset"]:
                selection_inds  = np.random.choice(len(sub_inds), size=cfg["label_file_subset"], replace=False).astype(np.int32)
                sub_inds = np.array(sub_inds)
                #print(selection_inds)
                sub_inds = sub_inds[selection_inds]
                

            if final_inds is None:
                 final_inds = np.array(sub_inds)
            else:
                final_inds = np.concatenate((final_inds, sub_inds), axis=0)
            #print(final_inds, "HERE")


        #print(final_inds)
        zarr.save(index_fname, final_inds)
    else:
        final_inds = zarr.load(index_fname)


    #print(final_inds)
    return final_inds


#Move from patch-level to pixel-level embeddigs
def rescale_embed(embed, image_shape, device, target, crop_info = None):
 
     ind = 0
     if embed.ndim > 3:
         ind = 1


     if int(math.sqrt(embed.shape[ind])) // 8 > 3:
  
         rescale_factor = int(math.sqrt(embed.shape[ind])) // 8


         while embed.shape[ind] % rescale_factor**2 > 0:
             rescale_factor = rescale_factor + 1

         ps = torch.nn.PixelShuffle(rescale_factor) #.to(device)
         embed = torch.from_numpy(embed).to(device)

         embed = ps(embed)
     else:
         embed = torch.from_numpy(embed).to(device)

     if embed.ndim < 4:
         embed = torch.unsqueeze(embed, dim=0)
     else:
         embed = torch.unsqueeze(torch.flatten(embed, start_dim=0, end_dim=1), dim=0)
         

     #Assumption is currently square images + tiles 
     #Adjusting to account for potential off-by-ones
     if embed.shape[-1] != image_shape:
         embed = F.interpolate(embed, size=(image_shape, image_shape), mode='nearest')
    
     #print(embed.shape, crop_info)
 
     if crop_info is not None:
         tmp = torch.zeros((embed.shape[0], embed.shape[1], crop_info[0][-2], crop_info[0][-1]))
         tmp[:,:,crop_info[1]:crop_info[1]+crop_info[3],crop_info[2]:crop_info[2]+crop_info[4]] = embed
         embed = tmp

         tmp2 = torch.zeros((1, crop_info[0][-2], crop_info[0][-1]))
         #print(tmp2.shape, target.shape)
         tmp2[:,crop_info[1]:crop_info[1]+crop_info[3],crop_info[2]:crop_info[2]+crop_info[4]] = target
         target = tmp2 

     target = target.flatten()

     embed = torch.permute(embed, (0,2,3,1)).flatten(start_dim=0, end_dim=2)
 
     if device == "cuda":
         embed = embed.detach().cpu().numpy()

 
     return embed, target

