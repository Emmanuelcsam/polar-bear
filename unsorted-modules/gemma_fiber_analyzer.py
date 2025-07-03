"""
Gemma 3 Vision-Language Model Integration for Fiber Optic Inspection
"""

from transformers import AutoProcessor, AutoModelForVision2Seq
import torch
import cv2
import numpy as np
import json
from typing import Dict, List, Tuple
from pathlib import Path

class GemmaFiberAnalyzer:
    """
    Uses Google Gemma 3 Vision model for fiber optic defect detection and analysis
    """
    
    def __init__(self, model_name: str = "google/gemma-3-vision", 
                 device: str = None):
        """
        Initialize Gemma 3 Vision model
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on (cuda/cpu/auto)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name, 
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
            device_map="auto" if device == "auto" else None
        )
        
        if device != "auto":
            self.model = self.model.to(self.device)
    
    def gemma_embedding(self, img_bgr: np.ndarray) -> np.ndarray:
        """
        Extract visual embeddings from image using Gemma vision encoder
        
        Args:
            img_bgr: BGR image from OpenCV
            
        Returns:
            1408-dimensional embedding vector
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.vision_model(**inputs, output_hidden_states=True)
        
        return outputs.pooler_output.squeeze().cpu().numpy()
    
    def detect_defects_multimodal(self, img_bgr: np.ndarray, 
                                  prompt: str = None) -> Dict:
        """
        Use Gemma 3 for defect detection with structured output
        
        Args:
            img_bgr: BGR image from OpenCV
            prompt: Custom prompt (uses default if None)
            
        Returns:
            Dictionary with detected defects and metadata
        """
        if prompt is None:
            prompt = """Analyze this fiber optic end-face image. 
            Return a JSON list with the following for each defect found:
            - type: (SCRATCH, PIT, CONTAMINATION, or OTHER)
            - bbox: [x, y, width, height] in pixels
            - severity: (LOW, MEDIUM, or HIGH)
            - zone: (CORE, CLADDING, or FERRULE)
            Only return the JSON, no other text."""
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # Process image and prompt
        inputs = self.processor(
            text=prompt,
            images=img_rgb,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate response
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,  # Low temperature for consistent JSON
                do_sample=True
            )
        
        # Decode response
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract JSON from response
        try:
            # Find JSON content
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start >= 0 and json_end > json_start:
                json_str = response[json_start:json_end]
                defects = json.loads(json_str)
            else:
                defects = []
        except json.JSONDecodeError:
            defects = []
        
        return {
            'defects': defects,
            'raw_response': response,
            'embedding': self.gemma_embedding(img_bgr)
        }
    
    def assess_fiber_quality(self, img_bgr: np.ndarray,
                           defects: List[Dict] = None) -> Dict:
        """
        Use Gemma to assess overall fiber quality
        
        Args:
            img_bgr: BGR image
            defects: Optional pre-detected defects
            
        Returns:
            Quality assessment dictionary
        """
        if defects is None:
            result = self.detect_defects_multimodal(img_bgr)
            defects = result.get('defects', [])
        
        prompt = f"""Given these defects found on a fiber optic end-face:
        {json.dumps(defects, indent=2)}
        
        Assess the overall quality. Return JSON with:
        - pass_fail: (PASS or FAIL)
        - quality_score: (0-100)
        - critical_issues: list of critical problems
        - recommendations: list of cleaning/repair recommendations
        """
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(
            text=prompt,
            images=img_rgb,
            return_tensors="pt"
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.3
            )
        
        response = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        try:
            # Extract JSON
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                assessment = json.loads(response[json_start:json_end])
            else:
                assessment = {
                    'pass_fail': 'UNKNOWN',
                    'quality_score': -1,
                    'error': 'Could not parse response'
                }
        except:
            assessment = {
                'pass_fail': 'UNKNOWN', 
                'quality_score': -1,
                'error': 'JSON parsing failed'
            }
        
        return assessment
    
    def compare_fibers(self, img1_bgr: np.ndarray, img2_bgr: np.ndarray) -> Dict:
        """
        Compare two fiber images using embeddings
        
        Args:
            img1_bgr, img2_bgr: Two fiber images to compare
            
        Returns:
            Similarity metrics
        """
        emb1 = self.gemma_embedding(img1_bgr)
        emb2 = self.gemma_embedding(img2_bgr)
        
        # Cosine similarity
        cos_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        
        # L2 distance
        l2_dist = np.linalg.norm(emb1 - emb2)
        
        return {
            'cosine_similarity': float(cos_sim),
            'l2_distance': float(l2_dist),
            'similar': cos_sim > 0.85  # Threshold for similarity
        }

# Integration helper for existing pipeline
class GemmaEnhancedAnalyzer:
    """
    Drop-in enhancement for existing fiber analysis pipeline
    """
    
    def __init__(self, use_gemma: bool = True):
        self.use_gemma = use_gemma
        self.gemma = GemmaFiberAnalyzer() if use_gemma else None
        
    def enhance_detection(self, image: np.ndarray, 
                         existing_results: Dict) -> Dict:
        """
        Enhance existing detection results with Gemma
        
        Args:
            image: Input image
            existing_results: Results from traditional detection
            
        Returns:
            Enhanced results dictionary
        """
        if not self.use_gemma or self.gemma is None:
            return existing_results
        
        # Get Gemma predictions
        gemma_results = self.gemma.detect_defects_multimodal(image)
        
        # Merge with existing results
        enhanced = existing_results.copy()
        enhanced['gemma_defects'] = gemma_results['defects']
        enhanced['gemma_embedding'] = gemma_results['embedding']
        
        # Get quality assessment
        quality = self.gemma.assess_fiber_quality(image, gemma_results['defects'])
        enhanced['gemma_assessment'] = quality
        
        # Override verdict if Gemma is more conservative
        if quality.get('pass_fail') == 'FAIL' and existing_results.get('verdict') == 'PASS':
            enhanced['verdict'] = 'FAIL'
            enhanced['verdict_reason'] = 'Gemma detected critical issues'
        
        return enhanced

# Example usage
if __name__ == "__main__":
    # Test the analyzer
    analyzer = GemmaFiberAnalyzer()
    
    # Load test image
    # img = cv2.imread("test_fiber.jpg")
    # results = analyzer.detect_defects_multimodal(img)
    # print(f"Detected defects: {results['defects']}")
    # 
    # quality = analyzer.assess_fiber_quality(img)
    # print(f"Quality assessment: {quality}")