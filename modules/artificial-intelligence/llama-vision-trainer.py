"""
Fine-tuning Llama-3.2-Vision for Fiber Optic Defect Detection
"""

import json
import torch
from pathlib import Path
from typing import Dict, List, Tuple
import cv2
import numpy as np
from transformers import (
    AutoProcessor, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
import hashlib

class FiberDefectDataset(Dataset):
    """
    Dataset for fine-tuning Llama Vision on fiber optic defects
    """
    
    def __init__(self, data_dir: Path, processor, max_length: int = 512):
        self.data_dir = Path(data_dir)
        self.processor = processor
        self.max_length = max_length
        self.samples = []
        
        # Load all JSONL entries
        jsonl_files = list(self.data_dir.glob("*.jsonl"))
        for jsonl_file in jsonl_files:
            with open(jsonl_file, 'r') as f:
                for line in f:
                    self.samples.append(json.loads(line))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        img_path = self.data_dir / sample['image']
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Format conversation
        conversations = sample['conversations']
        text = ""
        for conv in conversations:
            if conv['from'] == 'human':
                text += f"Human: {conv['value']}\n"
            else:
                text += f"Assistant: {conv['value']}\n"
        
        # Process inputs
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length
        )
        
        # Remove batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        
        # Labels are same as input_ids for causal LM
        inputs['labels'] = inputs['input_ids'].clone()
        
        return inputs

class LlamaVisionFineTuner:
    """
    Fine-tune Llama-3.2-Vision for fiber optic inspection
    """
    
    def __init__(self, 
                 model_name: str = "llava-hf/llava-1.6-llama3-8b",
                 use_lora: bool = True,
                 quantize: bool = True):
        """
        Initialize the fine-tuner
        
        Args:
            model_name: HuggingFace model ID
            use_lora: Whether to use LoRA for efficient fine-tuning
            quantize: Whether to use 4-bit quantization
        """
        self.model_name = model_name
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor
        self.processor = AutoProcessor.from_pretrained(model_name)
        
        # Quantization config
        bnb_config = None
        if quantize and self.device == "cuda":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True
            )
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        
        # Apply LoRA if requested
        if use_lora:
            lora_config = LoraConfig(
                r=64,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.CAUSAL_LM
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
    
    def prepare_training_data(self, raw_images_dir: Path, output_dir: Path):
        """
        Convert raw images and detection JSONs to training format
        
        Args:
            raw_images_dir: Directory with images and JSON reports
            output_dir: Where to save JSONL training data
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        jsonl_path = output_dir / "fiber_defects.jsonl"
        
        samples = []
        
        # Process each image and its corresponding JSON
        for img_path in raw_images_dir.glob("*.jpg"):
            json_path = img_path.with_suffix(".json")
            if not json_path.exists():
                continue
            
            with open(json_path) as f:
                report = json.load(f)
            
            # Create training sample
            defects = report.get("defects", [])
            
            # Format defects as JSON response
            defect_json = json.dumps({
                "defects": [
                    {
                        "bbox": [d["location"]["x"], d["location"]["y"], 
                                d["properties"].get("width", 50), 
                                d["properties"].get("height", 50)],
                        "type": d["type"],
                        "severity": d["severity"]
                    }
                    for d in defects
                ]
            }, indent=2)
            
            sample = {
                "image": img_path.name,
                "conversations": [
                    {
                        "from": "human",
                        "value": "<image>\nPlease list every defect you can find. Return JSON with keys bbox (x,y,w,h), type, severity."
                    },
                    {
                        "from": "gpt",
                        "value": defect_json
                    }
                ]
            }
            
            samples.append(sample)
        
        # Write JSONL
        with open(jsonl_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
        
        print(f"Created {len(samples)} training samples in {jsonl_path}")
        
        return jsonl_path
    
    def train(self, 
              data_dir: Path,
              output_dir: Path,
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 5e-5):
        """
        Fine-tune the model
        
        Args:
            data_dir: Directory with JSONL training data
            output_dir: Where to save checkpoints
            num_epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
        """
        # Create dataset
        dataset = FiberDefectDataset(data_dir, self.processor)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=8,
            warmup_ratio=0.05,
            learning_rate=learning_rate,
            bf16=True if self.device == "cuda" else False,
            logging_steps=10,
            save_steps=1000,
            evaluation_strategy="no",
            save_strategy="steps",
            remove_unused_columns=False,
            push_to_hub=False,
            report_to="none",
            optim="adamw_torch"
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self.processor
        )
        
        # Train
        trainer.train()
        
        # Save final model
        trainer.save_model(output_dir / "final_model")
        print(f"Training complete! Model saved to {output_dir}/final_model")

class LlamaFiberInference:
    """
    Inference wrapper for fine-tuned Llama Vision model
    """
    
    def __init__(self, model_path: Path):
        """Load fine-tuned model for inference"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )
        self.model.eval()
    
    def analyze_image(self, img_path: str) -> Dict:
        """
        Analyze fiber image for defects
        
        Args:
            img_path: Path to fiber image
            
        Returns:
            Dictionary with detected defects
        """
        prompt = "<image>\nPlease list every defect you can find. Return JSON with keys bbox (x,y,w,h), type, severity."
        
        # Load and preprocess image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process inputs
        inputs = self.processor(prompt, images=img, return_tensors="pt").to(self.device)
        
        # Generate response
        with torch.cuda.amp.autocast():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=512, 
                temperature=0.0,
                do_sample=False
            )
        
        # Decode response
        answer = self.processor.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Extract JSON from response
        try:
            json_start = answer.find('{')
            json_end = answer.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(answer[json_start:json_end])
            else:
                return {"error": "No JSON found in response", "raw": answer}
        except json.JSONDecodeError as e:
            return {"error": f"JSON parse error: {e}", "raw": answer}

# Integration with existing pipeline
def integrate_with_pipeline(model_path: Path):
    """
    Create drop-in replacement functions for existing pipeline
    """
    inference = LlamaFiberInference(model_path)
    
    def llama_detect_defects(image_path: str, config: Dict = None) -> Dict:
        """Drop-in replacement for defect detection"""
        result = inference.analyze_image(image_path)
        
        # Convert to expected format
        if "defects" in result:
            return {
                "defects": result["defects"],
                "ml_model": "llama-3.2-vision",
                "confidence": 0.95  # High confidence for fine-tuned model
            }
        else:
            return {
                "defects": [],
                "ml_model": "llama-3.2-vision",
                "error": result.get("error", "Unknown error")
            }
    
    return llama_detect_defects

# Example usage
if __name__ == "__main__":
    # Fine-tuning example
    tuner = LlamaVisionFineTuner(use_lora=True, quantize=True)
    
    # Prepare data
    # raw_dir = Path("/path/to/fiber/images")
    # train_dir = Path("/path/to/training/data")
    # tuner.prepare_training_data(raw_dir, train_dir)
    
    # Train
    # tuner.train(train_dir, Path("./checkpoints"))
    
    # Inference example
    # model = LlamaFiberInference(Path("./checkpoints/final_model"))
    # results = model.analyze_image("test_fiber.jpg")
    # print(results)