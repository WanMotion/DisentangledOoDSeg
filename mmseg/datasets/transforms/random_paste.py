import numpy as np
from mmcv.transforms.base import BaseTransform
from PIL import Image, ImageDraw
from mmseg.registry import TRANSFORMS
from scipy.spatial import ConvexHull
import cv2
from typing import Dict, List, Optional, Tuple, Union
from albumentations import RandomScale, Affine, HorizontalFlip, Compose, RandomCrop,ColorJitter
import random


@TRANSFORMS.register_module()
class RandomPaste(BaseTransform):

    def __init__(self, random_crop_ratio=0.5, overlap_threshold: float = 0.5, min_max_size=(32, 256),
                 max_num_polygons=10, ignore_index=255, new_class_index=19, transparency_ratios=(0.7, 1.0),paste_to_road_ratio=0.7):
        self.overlap_threshold = overlap_threshold
        self.min_max_size = min_max_size
        self.max_num_polygons = max_num_polygons
        self.ignore_index = ignore_index
        self.new_class_index = new_class_index
        self.transparency_ratios = transparency_ratios
        self.random_crop_ratio = random_crop_ratio
        self.paste_to_road_ratio=paste_to_road_ratio
        self.patch_pipeline = Compose([
            RandomScale((0.5,0.75), p=1.0),
            HorizontalFlip(),
            Affine(rotate=(-180, 180)),
            ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2, always_apply=True)
        ])

    # def _transform_1(self, results: dict) -> dict:
    #     img=results['img']
    #     gt_seg_map=results['gt_seg_map']
    #     h,w=img.shape[0],img.shape[1]
    #     cur_num_polygons=np.random.randint(self.max_num_polygons)
    #     for i in range(cur_num_polygons):
    #         height=np.random.randint(self.min_size,self.max_size)
    #         width=np.random.randint(self.min_size,self.max_size)
    #
    #         points_h=np.random.randint(0,height,(20,))
    #         points_w=np.random.randint(0,width,(20,))
    #         points=np.concatenate((np.expand_dims(points_w,1),np.expand_dims(points_h,1)),axis=1)
    #         hull=ConvexHull(points)
    #
    #         start_h=np.random.randint(0,h-height)
    #         start_w=np.random.randint(0,w-width)
    #         points[:,1]+=start_h
    #         points[:,0]+=start_w
    #         mask=Image.new('L',(w,h),0)
    #         mask_draw=ImageDraw.Draw(mask)
    #         polygons=points[hull.vertices].tolist()
    #         polygons=[(x,y) for x,y in polygons]
    #         mask_draw.polygon(polygons,fill=255)
    #         mask_arr=np.array(mask).astype(bool)
    #
    #         color = np.random.randint(0, 256, (3,))
    #         new_paste=np.full((h,w,3),color,dtype=np.float64)
    #         mean = 0
    #         variance = 0.5
    #         sigma = np.sqrt(variance)
    #         gaussian_noise = np.random.normal(mean, sigma, new_paste.shape)*5
    #         new_paste+=gaussian_noise
    #         img[mask_arr]=new_paste[mask_arr]
    #         gt_seg_map[mask_arr]=self.ignore_index
    #     results['img']=img.astype(np.uint8)
    #     results['gt_seg_map']=gt_seg_map.astype(np.uint8)
    #     return results
    # def transform_2(self,results: dict)->dict:
    #     img = results['img']
    #     gt_seg_map = results['gt_seg_map']
    #     h, w = img.shape[0], img.shape[1]
    #
    #     img_path=results['img_path']
    #     all_img_paths=os.listdir(os.path.dirname(img_path))
    #     cur_num_polygons = np.random.randint(self.max_num_polygons)
    #
    #     random_crop=RandomCrop(h,w)
    #     print("origin",img.shape)
    #     for i in range(cur_num_polygons):
    #         rand_idx=np.random.randint(0,len(all_img_paths))
    #         selected_img=all_img_paths[rand_idx]
    #         another_img_path=os.path.join(os.path.dirname(img_path),selected_img)
    #         ano_img=self.load_img({"img_path":another_img_path})["img"]
    #         ano_img=random_crop(image=ano_img)
    #
    #         height = np.random.randint(self.min_size, self.max_size)
    #         width = np.random.randint(self.min_size, self.max_size)
    #
    #         points_h = np.random.randint(0, height, (20,))
    #         points_w = np.random.randint(0, width, (20,))
    #         points = np.concatenate((np.expand_dims(points_w, 1), np.expand_dims(points_h, 1)), axis=1)
    #
    #         ano_h=np.random.randint(0,h-height)
    #         ano_w=np.random.randint(0,w-width)
    #
    #         # random scale
    #         max_ratio=min(h/height,w/width)-0.001
    #         ratio=np.random.uniform(self.scale_ratios[0],self.scale_ratios[1])
    #         ratio=min(ratio,max_ratio)
    #         points=(points*ratio).astype(np.int64)
    #         hull = ConvexHull(points)
    #         crop_ano_img=cv2.resize(ano_img[ano_h:ano_h+height,ano_w:ano_w+width,:],(int(width*ratio),int(height*ratio)))
    #         from_points = points.copy()
    #         from_polygons = from_points[hull.vertices].tolist()
    #         from_polygons = [(x, y) for x, y in from_polygons]
    #         from_mask_img = Image.new('L', (crop_ano_img.shape[1],crop_ano_img.shape[0]), 0)
    #         from_mask_draw = ImageDraw.Draw(from_mask_img)
    #         from_mask_draw.polygon(from_polygons, fill=255)
    #         from_mask_arr = np.array(from_mask_img).astype(bool)
    #
    #         this_h = np.random.randint(0, h - crop_ano_img.shape[0])
    #         this_w = np.random.randint(0, w - crop_ano_img.shape[1])
    #         to_points = points.copy()
    #         to_points[:, 1] += this_h
    #         to_points[:, 0] += this_w
    #         to_polygons = to_points[hull.vertices].tolist()
    #         to_polygons = [(x, y) for x, y in to_polygons]
    #         to_mask_img = Image.new('L', (w, h), 0)
    #         to_mask_draw = ImageDraw.Draw(to_mask_img)
    #         to_mask_draw.polygon(to_polygons, fill=255)
    #         to_mask_arr = np.array(to_mask_img).astype(bool)
    #
    #         overlay = img.copy()
    #         if to_mask_arr.sum()!=from_mask_arr.sum():
    #             continue
    #         overlay[to_mask_arr]=crop_ano_img[from_mask_arr]
    #
    #         trans_ratio = np.random.uniform(self.transparency_ratios[0], self.transparency_ratios[1])
    #         img=cv2.addWeighted(overlay,trans_ratio,img,1-trans_ratio,0)
    #         gt_seg_map[to_mask_arr]=self.new_class_index
    #
    #     results['img'] = img.astype(np.uint8)
    #     results['gt_seg_map'] = gt_seg_map.astype(np.uint8)
    #
    #     return results
    # def transform_3(self,results: dict)->dict:
    #     img = results['img']
    #     gt_seg_map = results['gt_seg_map']
    #     h, w = img.shape[0], img.shape[1]
    #     cur_num_polygons = np.random.randint(self.max_num_polygons)
    #
    #     random_crop=RandomCrop(h,w)
    #     for i in range(cur_num_polygons):
    #         # load random img
    #         rand_idx=np.random.randint(0,len(self.all_img_paths))
    #         selected_img=self.all_img_paths[rand_idx]
    #         ano_img=self.load_img({"img_path":selected_img})["img"]
    #         ano_img=random_crop(image=ano_img)["image"]
    #         # crop size
    #         height = np.random.randint(self.min_size, self.max_size)
    #         width = np.random.randint(self.min_size, self.max_size)
    #         # crop loc
    #         ano_h=np.random.randint(0,h-height)
    #         ano_w=np.random.randint(0,w-width)
    #         crop_img=ano_img[ano_h:ano_h+height,ano_w:ano_w+width,:]
    #         # random scale
    #         max_ratio=min(h/height,w/width)-0.001
    #         ratio=np.random.uniform(self.scale_ratios[0],self.scale_ratios[1])
    #         ratio=min(ratio,max_ratio)
    #         scale_trans=Affine(scale=ratio)
    #         crop_scale_img=scale_trans(image=crop_img)["image"]
    #         # poly mask
    #         points_h = np.random.randint(0, crop_scale_img.shape[0], (20,))
    #         points_w = np.random.randint(0, crop_scale_img.shape[1], (20,))
    #         points = np.concatenate((np.expand_dims(points_w, 1), np.expand_dims(points_h, 1)), axis=1)
    #         hull = ConvexHull(points)
    #
    #         from_points = points.copy()
    #         from_polygons = from_points[hull.vertices].tolist()
    #         from_polygons = [(x, y) for x, y in from_polygons]
    #         from_mask_img = Image.new('L', (crop_scale_img.shape[1],crop_scale_img.shape[0]), 0)
    #         from_mask_draw = ImageDraw.Draw(from_mask_img)
    #         from_mask_draw.polygon(from_polygons, fill=255)
    #         from_mask_arr = np.array(from_mask_img).astype(bool)
    #
    #         this_h = np.random.randint(0, h - crop_scale_img.shape[0])
    #         this_w = np.random.randint(0, w - crop_scale_img.shape[1])
    #         to_points = points.copy()
    #         to_points[:, 1] += this_h
    #         to_points[:, 0] += this_w
    #         to_polygons = to_points[hull.vertices].tolist()
    #         to_polygons = [(x, y) for x, y in to_polygons]
    #         to_mask_img = Image.new('L', (w, h), 0)
    #         to_mask_draw = ImageDraw.Draw(to_mask_img)
    #         to_mask_draw.polygon(to_polygons, fill=255)
    #         to_mask_arr = np.array(to_mask_img).astype(bool)
    #
    #
    #         overlay = img.copy()
    #         if to_mask_arr.sum()!=from_mask_arr.sum():
    #             continue
    #         overlay[to_mask_arr]=crop_scale_img[from_mask_arr]
    #
    #         # transparency
    #         trans_ratio = np.random.uniform(self.transparency_ratios[0], self.transparency_ratios[1])
    #         img=cv2.addWeighted(overlay,trans_ratio,img,1-trans_ratio,0)
    #         gt_seg_map[to_mask_arr]=self.new_class_index
    #
    #     results['img'] = img.astype(np.uint8)
    #     results['gt_seg_map'] = gt_seg_map
    #
    #     return results

    def get_indices(self, dataset) -> list:

        indices = [np.random.randint(0, len(dataset)) for _ in range(self.max_num_polygons)]
        return indices

    def random_crop(self,img:np.ndarray,mask:np.ndarray,h:int,w:int,cut_max:float=0.5):
        img_h,img_w=img.shape[0],img.shape[1]
        mask_c=mask!=0
        mask_c[img_h-h:,:]=False
        mask_c[:,img_w-w:]=False
        points=np.where(mask_c)
        length=points[0].shape[0]

        final_h,final_w=0,0
        for k in range(100):
            idx=random.randint(0,length-1)
            pos_h,pos_w=points[0][idx],points[1][idx]
            cut=mask[pos_h:pos_h+h,pos_w:pos_w+w]
            ids=np.unique(cut)
            flag=False
            for i in ids:
                if (cut==i).sum()/(h*w)>cut_max:
                    flag=True
                    break
            if flag:
                continue
            else:
                final_h,final_w=pos_h,pos_w
                break

        img_crop=img.copy()[final_h:final_h+h,final_w:final_w+w,:]
        mask_crop=mask.copy()[final_h:final_h+h,final_w:final_w+w]

        return img_crop,mask_crop


    def transform(self,
                  results: Dict) -> Optional[Union[Dict, Tuple[List, List]]]:
        img = results['img']
        img_crop = img.copy()
        gt_seg_map = results['gt_seg_map']

        other_results = results['mix_results']
        h, w = img.shape[0], img.shape[1]

        from_masks=[]
        from_patches=[]
        from_gts=[]

        cur_num_polygons = self.max_num_polygons #np.random.randint(self.max_num_polygons)
        count = 0
        for i in range(cur_num_polygons):
            rand_h = random.randint(self.min_max_size[0], self.min_max_size[1])
            rand_w = rand_h
            patch = other_results[i]
            patch_trans = self.patch_pipeline(image=cv2.cvtColor(patch['img'],cv2.COLOR_BGR2RGB), mask=patch['gt_seg_map'])

            img_cvt=cv2.cvtColor(patch_trans['image'],cv2.COLOR_RGB2BGR)
            from_patches.append(img_cvt.copy())
            from_gts.append(patch_trans['mask'].copy())
            patch_img,patch_gt = self.random_crop(img_cvt,patch_trans['mask'],rand_h,rand_w)

            from_mask_img = Image.new('L', (patch_img.shape[1], patch_img.shape[0]), 0)
            from_mask_draw = ImageDraw.Draw(from_mask_img)
            if random.random() < 0.7:
                points_h = np.random.randint(0, patch_img.shape[0], (10,))
                points_w = np.random.randint(0, patch_img.shape[1], (10,))
                points = np.concatenate((np.expand_dims(points_w, 1), np.expand_dims(points_h, 1)), axis=1)
                hull = ConvexHull(points)

                from_points = points.copy()
                from_polygons = from_points[hull.vertices].tolist()
                from_polygons = [(x, y) for x, y in from_polygons]
                from_mask_draw.polygon(from_polygons, fill=255)

            else:
                p = random.random()
                if p < 0.3:
                    r=random.random()*0.7+0.3
                    if random.random()<0.5:
                        from_mask_draw.ellipse((0, 0, patch_img.shape[1], int(patch_img.shape[0]*r)), fill=255)
                    else:
                        from_mask_draw.ellipse((0, 0, int(patch_img.shape[1]*r), patch_img.shape[0]), fill=255)
                elif p < 0.66:
                    min_h_w = min(patch_img.shape[1], patch_img.shape[0])
                    from_mask_draw.ellipse((0, 0, min_h_w, min_h_w), fill=255)
                else:
                    from_mask_draw.rectangle((0, 0, patch_img.shape[1], patch_img.shape[0]), fill=255)

            from_mask_arr = np.array(from_mask_img).astype(bool)
            from_masks.append(from_mask_arr)


            m_to_paste=gt_seg_map==0
            m_to_paste[h-patch_img.shape[0]:,:]=False
            m_to_paste[:,w-patch_img.shape[1]:]=False
            points=np.where(m_to_paste)
            def _get_paste_pos(points,h,w,p_h,p_w,anywhere):
                if anywhere:
                    t_h,t_w=np.random.randint(0, h - p_h),np.random.randint(0, w - p_w)
                else:
                    length=points[0].shape[0]
                    if length==0:
                        t_h, t_w = np.random.randint(0, h - p_h), np.random.randint(0, w - p_w)
                    else:
                        idx=random.randint(0,length-1)
                        t_h,t_w=points[0][idx],points[1][idx]
                return t_h,t_w

            _r=random.random()>self.paste_to_road_ratio

            this_h, this_w=_get_paste_pos(points,h,w,patch_img.shape[0],patch_img.shape[1],_r)
            to_mask_img = Image.new('L', (w, h), 0)
            to_mask_img.paste(from_mask_img, (this_w, this_h))
            to_mask_arr = np.array(to_mask_img).astype(bool)
            if random.random() < self.random_crop_ratio:
                k = 0
                while k < 10:
                    k += 1
                    if (gt_seg_map[to_mask_arr] == patch_gt[
                        from_mask_arr]).sum() / to_mask_arr.sum() > self.overlap_threshold:
                        this_h, this_w=_get_paste_pos(points,h,w,patch_img.shape[0],patch_img.shape[1],_r)
                        to_mask_img = Image.new('L', (w, h), 0)
                        to_mask_img.paste(from_mask_img, (this_w, this_h))
                        to_mask_arr = np.array(to_mask_img).astype(bool)
                        continue
                    else:
                        break
            else:
                if random.random() > 0.7:
                    patch_img = (np.random.random(patch_img.shape) * 255).astype(np.uint8)
                else:
                    color = (255 * np.random.rand(3))[None, None, :]
                    noise = (25 * np.random.rand(*patch_img.shape) - 12)
                    patch_img = (np.ones(patch_img.shape) * color + noise).astype(np.uint8)

            overlay = img.copy()
            # if to_mask_arr.sum()!=from_mask_arr.sum():
            #     continue
            overlay[to_mask_arr] = patch_img[from_mask_arr]
            img_crop[to_mask_arr] = 0

            # transparency
            trans_ratio = np.random.uniform(self.transparency_ratios[0], self.transparency_ratios[1])
            img = cv2.addWeighted(overlay, trans_ratio, img, 1 - trans_ratio, 0)

            gt_seg_map[to_mask_arr] = self.new_class_index#+count
            count += 1

        results['img'] = img.astype(np.uint8)
        results['gt_seg_map'] = gt_seg_map

        # deal patches paste to overlap
        m_idx=random.randint(0,len(from_masks)-1)
        _h,_w=from_masks[m_idx].shape[0],from_masks[m_idx].shape[1]
        p_idx_1=random.randint(0,len(from_patches)-1)
        patch_to_paste_to_overlap_1,_=self.random_crop(from_patches[p_idx_1],from_gts[p_idx_1],_h,_w)
        p_idx_2=random.randint(0,len(from_patches)-1)
        patch_to_paste_to_overlap_2,_=self.random_crop(from_patches[p_idx_2],from_gts[p_idx_2],_h,_w)
        results['paste_to_overlap']={
            'mask':from_masks[m_idx],
            'patch_1':patch_to_paste_to_overlap_1,
            'patch_2':patch_to_paste_to_overlap_2
        }

        # img=Image.fromarray(results['img'].astype(np.uint8))
        # img.save('trans.png')
        # img_t=Image.fromarray(results['gt_seg_map'].astype(np.uint8))
        # img_t.save('trans_t.png')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_size={self.min_size},max_size={self.max_size},max_num_polygons={self.max_num_polygons},ignore_index={self.ignore_index},new_class_index={self.new_class_index})'
        return repr_str

