import google.generativeai as genai
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import torch
from dotenv import load_dotenv
import os
import re

load_dotenv()

class LLMComparator:
    def __init__(self):
        # Initialize GIT model for captioning
        self.git_processor = AutoProcessor.from_pretrained("microsoft/git-base-coco")
        self.git_model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-coco")
        
        # Initialize Gemini
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel('gemini-2.5-flash')
    
    def generate_caption(self, image_path):
        image = Image.open(image_path)
        inputs = self.git_processor(images=image, return_tensors="pt")
        
        generated_ids = self.git_model.generate(
            pixel_values=inputs.pixel_values,
            max_length=50
        )
        caption = self.git_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return caption
    
    def analyze_with_gemini(self, image_path, text, caption):
        image = Image.open(image_path)
        
        prompt = f"""
        Analyze how well this text matches the image content:
        
        User Provided Text: {text}
        Image Caption: {caption}
        
        Perform a detailed analysis:
        1. Identify matching elements between the text and image (be specific)
        2. Point out any discrepancies or missing elements (be specific)
        3. Provide a similarity score (0-100%) with justification
        4. Suggest improvements to make the text better match the image
        
        Format your response EXACTLY as follows:
        
        ### Matching Elements:
        [list specific matching elements]
        
        ### Discrepancies:
        [list specific discrepancies]
        
        ### Similarity Score:
        [number between 0-100]%
        
        ### Justification:
        [explanation of score]
        
        ### Suggestions:
        [specific suggestions for improvement]
        """
        
        try:
            # Generate content
            response = self.gemini_model.generate_content([prompt, image])
            
            # Try to get text response safely
            if hasattr(response, 'text'):
                response_text = response.text
            elif hasattr(response, 'result'):
                response_text = response.result
            elif hasattr(response, 'content'):
                content = getattr(response, 'content')

                # Handle both list and string types
                if isinstance(content, list) and len(content) > 0:
                    first_item = content[0]
                else:
                    first_item = content  # fallback for string or non-list

                if hasattr(first_item, 'text'):
                    response_text = first_item.text
                elif hasattr(first_item, 'parts'):
                    parts = getattr(first_item, 'parts', [])
                    text_parts = []
                    for part in parts:
                        if hasattr(part, 'text'):
                            text_parts.append(str(part.text))
                        elif hasattr(part, 'content'):
                            text_parts.append(str(part.content))
                    response_text = '\n'.join(text_parts)
                else:
                    response_text = str(first_item)
            else:
                response_text = str(response)
            
            # Ensure we have a valid string
            if not isinstance(response_text, str):
                response_text = str(response_text)
            
            # Basic validation
            if not response_text.strip():
                return {
                    "error": "Empty response from Gemini",
                    "matching_elements": "",
                    "discrepancies": "",
                    "score": 0,
                    "justification": "",
                    "suggestions": ""
                }
            
            # Log the response for debugging
            print(f"Gemini Response Text: {response_text}")
            
            return self._parse_gemini_response(response_text)
        except Exception as e:
            return {
                "error": f"Gemini API error: {str(e)}",
                "matching_elements": "",
                "discrepancies": "",
                "score": 0,
                "justification": "",
                "suggestions": ""
            }
    
    def _parse_gemini_response(self, response_text):
        # Parse the structured response
        result = {
            "matching_elements": "",
            "discrepancies": "",
            "score": 0,
            "justification": "",
            "suggestions": ""
        }
        
        try:
            # Print the raw response for debugging
            print(f"Raw Response Text: {repr(response_text)}")
            
            # Convert to string if not already
            if not isinstance(response_text, str):
                response_text = str(response_text)
            
            # Extract matching elements
            match_section = re.search(r'### Matching Elements:\n(.*?)(\n\n###|\n###|\Z)', response_text, re.DOTALL)
            if match_section:
                result["matching_elements"] = match_section.group(1).strip()
            
            # Extract discrepancies
            disc_section = re.search(r'### Discrepancies:\n(.*?)(\n\n###|\n###|\Z)', response_text, re.DOTALL)
            if disc_section:
                result["discrepancies"] = disc_section.group(1).strip()
            
            # Extract score
            score_match = re.search(r'### Similarity Score:\n(\d+)%', response_text)
            if score_match:
                result["score"] = int(score_match.group(1))
            
            # Extract justification
            just_section = re.search(r'### Justification:\n(.*?)(\n\n###|\n###|\Z)', response_text, re.DOTALL)
            if just_section:
                result["justification"] = just_section.group(1).strip()
            
            # Extract suggestions
            sugg_section = re.search(r'### Suggestions:\n(.*?)(\n\n###|\n###|\Z|$)', response_text, re.DOTALL)
            if sugg_section:
                result["suggestions"] = sugg_section.group(1).strip()
            
        except Exception as e:
            print(f"Error parsing response: {str(e)}")
            print(f"Response text was: {repr(response_text)}")
            result["error"] = f"Response parsing error: {str(e)}"
        
        return result
