import gradio as gr
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import dotenv
import os
from PIL import Image
from pathlib import Path
from typing import List

dotenv.load_dotenv()
hf_token = os.getenv("HF_TOKEN")


class DataGenerator:
    def __init__(self, image_pathes, batch_size=512):
        self.image_pathes = image_pathes
        self.batch_size = batch_size

    def generate(self):
        for i in range(0, len(self.image_pathes), self.batch_size):
            images = [Image.open(image_path) for image_path in self.image_pathes[i:i+self.batch_size]]
            yield images
    
    

class ImageSearch:
    def __init__(self, 
                 image_dir: str, 
                 model_name: str = "line-corporation/clip-japanese-base", 
                 image_extension: List[str] = ["jpg", "jpeg", "png", "JPG", "JPEG", "PNG"]
                 ):
        self.img_dir = image_dir
        path = Path(image_dir)
        self.image_pathes = []
        for ext in image_extension:
            self.image_pathes.extend(list(path.glob(f"**/*.{ext}")))
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to("cuda")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.processor = AutoImageProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.datagenerator = DataGenerator(self.image_pathes)
        self.image_embedded = self.preprocess()
    
    def preprocess(self):
        image_embedded = []
        with torch.no_grad():
            for images in self.datagenerator.generate():
                inputs = self.processor(images=images, return_tensors="pt")
                inputs = {k: v.cuda() for k, v in inputs.items()}
                features = self.model.get_image_features(**inputs)
                image_embedded.append(features)
        image_embedded = torch.cat(image_embedded, dim=0).cuda()
        return image_embedded
    
    def search(self, positive_query: str, negative_query: str, top_k: int = 1, delta: float = 1.0):
        with torch.no_grad():
            positive_text_input = self.tokenizer([positive_query]).to("cuda")
            negative_text_input = self.tokenizer([negative_query]).to("cuda")
            positive_text_features = self.model.get_text_features(**positive_text_input)
            negative_text_features = self.model.get_text_features(**negative_text_input)
            image_embedded = self.image_embedded
            positive_score = (image_embedded @ positive_text_features.T).cpu().numpy().squeeze()
            negative_score = (image_embedded @ negative_text_features.T).cpu().numpy().squeeze()
            if negative_query == "" or negative_query == None:
                similarity = positive_score
            else:
                similarity = positive_score - delta * negative_score
            sorted_indices = similarity.argsort()[::-1]
        return [(Image.open(self.image_pathes[i]), f"sim: {similarity[i]}") for i in sorted_indices[:top_k]]


if __name__ == "__main__":
    searcher = ImageSearch("./Takeout")
    
    demo = gr.Interface(
        fn=searcher.search,
        inputs=[
            gr.Textbox(label="検索クエリ", placeholder="検索したい画像のキーワードを入力してください"),
            gr.Textbox(label="ネガティブクエリ", placeholder="検索したくない画像のキーワードを入力してください"),
            gr.Number(value=20, label="検索結果数", minimum=1, maximum=30),
            gr.Slider(value=0.3, label="ネガティブクエリの寄与度", minimum=0.0, maximum=5.0, step=0.1)
            ],
        outputs=gr.Gallery(label="検索結果", rows=4, columns=4),
    )
    demo.launch()
    
