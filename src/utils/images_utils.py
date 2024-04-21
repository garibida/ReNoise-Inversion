from PIL import Image
import os
import torch

def read_images_in_path(path, size = (512,512)):
    image_paths = []
    for filename in os.listdir(path):
        if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".jpeg"):
            image_path = os.path.join(path, filename)
            image_paths.append(image_path)
    image_paths = sorted(image_paths)
    return [Image.open(image_path).convert("RGB").resize(size) for image_path in image_paths]

def concatenate_images(image_lists, return_list = False):
    num_rows = len(image_lists[0])
    num_columns = len(image_lists)
    image_width = image_lists[0][0].width
    image_height = image_lists[0][0].height

    grid_width = num_columns * image_width
    grid_height = num_rows * image_height if not return_list else image_height
    if not return_list:
        grid_image = [Image.new('RGB', (grid_width, grid_height))]
    else:
        grid_image = [Image.new('RGB', (grid_width, grid_height)) for i in range(num_rows)]

    for i in range(num_rows):
        row_index = i if return_list else 0
        for j in range(num_columns):
            image = image_lists[j][i]
            x_offset = j * image_width
            y_offset = i * image_height if not return_list else 0
            grid_image[row_index].paste(image, (x_offset, y_offset))

    return grid_image if return_list else grid_image[0]

def concatenate_images_single(image_lists):
    num_columns = len(image_lists)
    image_width = image_lists[0].width
    image_height = image_lists[0].height

    grid_width = num_columns * image_width
    grid_height = image_height
    grid_image = Image.new('RGB', (grid_width, grid_height))

    for j in range(num_columns):
        image = image_lists[j]
        x_offset = j * image_width
        y_offset = 0
        grid_image.paste(image, (x_offset, y_offset))

    return grid_image

def get_captions_for_images(images, device):
    from transformers import Blip2Processor, Blip2ForConditionalGeneration

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        "Salesforce/blip2-opt-2.7b", load_in_8bit=True, device_map={"": 0}, torch_dtype=torch.float16
    )  # doctest: +IGNORE_RESULT

    res = []
    
    for image in images:
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

        generated_ids = model.generate(**inputs)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        res.append(generated_text)

    del processor
    del model
    
    return res