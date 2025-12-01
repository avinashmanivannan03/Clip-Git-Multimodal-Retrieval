import torch
from PIL import Image
from sentence_transformers import SentenceTransformer, util

class CLIPComparator:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer('clip-ViT-B-32', device=self.device)
        self.model.eval()

    def get_image_embedding(self, image_path):
        """Get proper CLIP embedding for an image"""
        try:
            img = Image.open(image_path).convert('RGB')
            with torch.no_grad():
                img_emb = self.model.encode(img, convert_to_tensor=True, show_progress_bar=False)
            return img_emb
        except Exception as e:
            raise ValueError(f"Image processing error: {str(e)}")

    def get_text_embedding(self, text):
        """Get proper CLIP embedding for text"""
        if not isinstance(text, str):
            raise ValueError("Text must be a string")

        text = text.strip().lower()
        if not text.endswith('.'):
            text += '.'

        try:
            with torch.no_grad():
                text_emb = self.model.encode(text, convert_to_tensor=True, show_progress_bar=False)
            return text_emb
        except Exception as e:
            raise ValueError(f"Text processing error: {str(e)}")

    def calculate_similarity(self, image_path, text):
        """Calculate sigmoid-style similarity score between 0-100%"""
        try:
            img_emb = self.get_image_embedding(image_path)
            text_emb = self.get_text_embedding(text)

            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            cos_sim = torch.matmul(img_emb, text_emb.T).item()

            # Sigmoid-style scaling: sharper separation between matches and mismatches
            score = 100 * (1.0 - 1.0 / (1.0 + 10 * (cos_sim - 0.7)))
            return max(0, min(100, round(score, 2)))
        except Exception as e:
            print(f"Similarity calculation error: {str(e)}")
            return 0.0

    def detailed_analysis(self, image_path, text):
        """Provide accurate detailed analysis with word-level scores"""
        try:
            img_emb = self.get_image_embedding(image_path)
            text_emb = self.get_text_embedding(text)

            img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)

            overall_score = self.calculate_similarity(image_path, text)

            words = [w for w in text.lower().split() if len(w) > 2 and w.isalpha()]
            word_scores = []

            if words:
                word_phrases = [f"a photo of {word}" for word in words]
                word_embs = self.model.encode(word_phrases, convert_to_tensor=True)
                word_embs = word_embs / word_embs.norm(dim=-1, keepdim=True)

                for word, word_emb in zip(words, word_embs):
                    word_sim = torch.matmul(img_emb, word_emb.T).item()
                    word_score = 100 * (1.0 - 1.0 / (1.0 + 10 * (word_sim - 0.7)))
                    word_scores.append((word, round(max(0, min(100, word_score)), 1)))

            word_scores_sorted = sorted(word_scores, key=lambda x: x[1])

            return {
                "overall_score": overall_score,
                "word_scores": word_scores,
                "least_matching": word_scores_sorted[:3],
                "most_matching": word_scores_sorted[-3:][::-1],
                "matching_regions": self._identify_matching_regions(image_path, text),
                "caption_note": "Token-level similarity (higher means better image-text alignment)"
            }

        except Exception as e:
            print(f"Analysis error: {str(e)}")
            return {
                "error": str(e),
                "overall_score": 0,
                "word_scores": [],
                "least_matching": [],
                "most_matching": [],
                "matching_regions": [],
                "caption_note": "Token-level similarity (higher means better image-text alignment)"
            }

    def _identify_matching_regions(self, image_path, text):
        return ["center", "foreground"]  # Placeholder
