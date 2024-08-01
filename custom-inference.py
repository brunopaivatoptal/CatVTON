from diffusers.image_processor import VaeImageProcessor
from torch.utils.data import Dataset, DataLoader
from inferenceConfig import inferenceConfig
from model.pipeline import CatVTONPipeline
from PIL import Image, ImageFilter
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch
import os

class InferenceDataset(Dataset):
    def __init__(self, args):
        self.args = args
    
        self.vae_processor = VaeImageProcessor(vae_scale_factor=8) 
        self.mask_processor = VaeImageProcessor(vae_scale_factor=8, do_normalize=False, do_binarize=True, do_convert_grayscale=True) 
        self.data = self.load_data()
    
    def load_data(self):
        return []
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        data = self.data[idx]
        person, cloth, mask = [Image.open(data[key]) for key in ['person', 'cloth', 'mask']]
        return {
            'index': idx,
            'person_name': data['person_name'],
            'person': self.vae_processor.preprocess(person, self.args.height, self.args.width)[0],
            'cloth': self.vae_processor.preprocess(cloth, self.args.height, self.args.width)[0],
            'mask': self.mask_processor.preprocess(mask, self.args.height, self.args.width)[0]
        }

class PxVtonDataset(InferenceDataset):
    def load_data(self):
        data = []
        data_root = [x for x in self.args.data_root_path if os.path.exists(x)][0]
        data_root = Path(data_root)
        allFiles = [data_root/x for x in os.listdir(data_root)]
        allFilesLoop = [allFiles[-1]] + allFiles[:-1]
        for reference, garment in zip(allFiles, allFilesLoop):
            name = os.path.split(reference)[-1]
            data.append({
                'person_name': name,
                'person': reference/"person"/os.listdir(reference/"person")[0],
                'cloth': garment/"garment"/os.listdir(garment/"garment")[0],
                'mask': reference/"cloth_agnostic_mask"/os.listdir(reference/"cloth_agnostic_mask")[0],
            })
        return data


def repaint(person, mask, result):
    _, h = result.size
    kernal_size = h // 50
    if kernal_size % 2 == 0:
        kernal_size += 1
    mask = mask.filter(ImageFilter.GaussianBlur(kernal_size))
    person_np = np.array(person)
    result_np = np.array(result)
    mask_np = np.array(mask) / 255
    repaint_result = person_np * (1 - mask_np) + result_np * mask_np
    repaint_result = Image.fromarray(repaint_result.astype(np.uint8))
    return repaint_result

def to_pil_image(images):
    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.cpu().permute(0, 2, 3, 1).float().numpy()
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]
    return pil_images

@torch.no_grad()
def main():
    args = inferenceConfig()
    # Pipeline
    pipeline = CatVTONPipeline(
        attn_ckpt_version="mix",
        attn_ckpt=args.resume_path,
        base_ckpt=args.base_model_path,
        weight_dtype={
            "no": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[args.mixed_precision],
        device="cuda",
        skip_safety_check=True
    )
    # Dataset
    dataset = PxVtonDataset(args)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.dataloader_num_workers
    )
    # Inference
    generator = torch.Generator(device='cuda').manual_seed(args.seed)
    args.output_dir = os.path.join(args.output_dir, f"{args.dataset_name}-{args.height}", "paired" if args.eval_pair else "unpaired")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    for batch in tqdm(dataloader):
        person_images = batch['person']
        cloth_images = batch['cloth']
        masks = batch['mask']
        results = pipeline(
            person_images,
            cloth_images,
            masks,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            generator=generator,
        )
        
        if args.concat_eval_results or args.repaint:
            person_images = to_pil_image(person_images)
            cloth_images = to_pil_image(cloth_images)
            masks = to_pil_image(masks)
        for i, result in enumerate(results):
            person_name = batch['person_name'][i]
            output_path = os.path.join(args.output_dir, person_name)
            if not os.path.exists(os.path.dirname(output_path)):
                os.makedirs(os.path.dirname(output_path))
            if args.repaint:
                person_path, mask_path = dataset.data[batch['index'][i]]['person'], dataset.data[batch['index'][i]]['mask']
                person_image= Image.open(person_path).resize(result.size, Image.LANCZOS)
                mask = Image.open(mask_path).resize(result.size, Image.NEAREST)
                result = repaint(person_image, mask, result)
            if args.concat_eval_results:
                w, h = result.size
                concated_result = Image.new('RGB', (w*3, h))
                concated_result.paste(person_images[i], (0, 0))
                concated_result.paste(cloth_images[i], (w, 0))  
                concated_result.paste(result, (w*2, 0))
                result = concated_result
            result.save(output_path + ".png")

if __name__ == "__main__":
    main()
    