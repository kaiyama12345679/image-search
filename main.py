import gradio as gr
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import dotenv
import os
from PIL import Image

dotenv.load_dotenv()
hf_token = os.getenv("HF_TOKEN")


class DataGenerator:
    def __init__(self, image_pathes, batch_size=256):
        self.image_pathes = image_pathes
        self.batch_size = batch_size

    def generate(self):
        for i in range(0, len(self.image_pathes), self.batch_size):
            images = [Image.open(image_path) for image_path in self.image_pathes[i:i+self.batch_size]]
            yield images
    
    

class ImageSearch:
    def __init__(self, image_dir: str, model_name: str = "line-corporation/clip-japanese-base"):
        self.img_dir = image_dir
        self.image_pathes = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)][:10000]
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
                inputs = {k: v.to("cuda") for k, v in inputs.items()}
                features = self.model.get_image_features(**inputs)
                image_embedded.append(features)
        image_embedded = torch.cat(image_embedded, dim=0).cuda()
        return image_embedded
    
    def search(self, query: str, top_k: int = 6):
        with torch.no_grad():
            text_input = self.tokenizer([query]).to("cuda")
            text_features = self.model.get_text_features(**text_input)
            image_embedded = self.image_embedded
            similarity = (image_embedded @ text_features.T).cpu().numpy().squeeze()
            sorted_indices = similarity.argsort()[::-1]
        return [(Image.open(self.image_pathes[i]), f"sim: {similarity[i]}") for i in sorted_indices[:top_k]]


if __name__ == "__main__":
    searcher = ImageSearch("animefaces256cleaner")
    
    demo = gr.Interface(
        fn=searcher.search,
        inputs=gr.Textbox(label="検索クエリ", placeholder="検索したい画像のキーワードを入力してください"),
        outputs=gr.Gallery(label="検索結果", rows=2, columns=3),
        live=True,
    )
    demo.launch()
    
