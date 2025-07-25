# app.py
# Modern web interface for real-time fiber optic analysis

import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import logging

from config import ConfigManager
from model import load_model
from utils import preprocess_image_for_inference, create_segmentation_overlay, create_anomaly_heatmap

class FiberOpticAnalysisApp:
    """Web application for fiber optic analysis."""
    
    def __init__(self, config_path="config.yaml", checkpoint_path=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Load configuration
        try:
            self.config_manager = ConfigManager(config_path)
            self.config = self.config_manager.config
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
        
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.logger.info(f"Using device: {self.device}")
        
        # Use checkpoint from config if not provided
        if checkpoint_path is None:
            checkpoint_path = self.config.webapp.default_checkpoint
        
        # Load model
        self.model = self._load_model(checkpoint_path)
        
        # Setup Gradio interface if available
        if GRADIO_AVAILABLE:
            self.interface = self._create_gradio_interface()
        else:
            self.logger.warning("Gradio not available. Install with: pip install gradio")
            self.interface = None
    
    def _load_model(self, checkpoint_path=None):
        """Load the trained model."""
        if checkpoint_path is None:
            # Try to find the best checkpoint
            checkpoint_dir = Path(self.config.system.checkpoints_path)
            best_checkpoint = checkpoint_dir / "best_model.pth"
            if best_checkpoint.exists():
                checkpoint_path = best_checkpoint
            else:
                # Look for the latest epoch checkpoint
                epoch_checkpoints = list(checkpoint_dir.glob("epoch_*.pth"))
                if epoch_checkpoints:
                    checkpoint_path = max(epoch_checkpoints, key=lambda x: int(x.stem.split('_')[-1]))
                else:
                    self.logger.warning("No checkpoint found. Using untrained model.")
                    checkpoint_path = None
        
        try:
            model = load_model(self.config, checkpoint_path, self.device)
            model.eval()
            self.logger.info(f"Model loaded from: {checkpoint_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            # Create a fresh model as fallback
            from model import create_model
            model = create_model(self.config).to(self.device)
            model.eval()
            self.logger.warning("Using untrained model")
        
        return model
    
    def analyze_image(self, image, coeff_A=None, coeff_B=None, coeff_C=None):
        """
        Analyze a single fiber optic image.
        
        Args:
            image: Input image (PIL Image)
            coeff_A: Coefficient for classification confidence
            coeff_B: Coefficient for anomaly score
            coeff_C: Coefficient for embedding similarity
            
        Returns:
            Dictionary containing analysis results
        """
        if image is None:
            return {
                'error': 'No image provided',
                'segmentation': None,
                'anomaly_heatmap': None,
                'results_text': 'Please upload an image.',
                'class_probabilities': {}
            }
        
        try:
            # Preprocess image
            input_tensor = preprocess_image_for_inference(
                image, target_size=self.config.data.image_size
            ).to(self.device)
            
            # Set equation coefficients
            coeffs = {
                'A': coeff_A or self.config.equation.coefficients.A,
                'B': coeff_B or self.config.equation.coefficients.B,
                'C': coeff_C or self.config.equation.coefficients.C
            }
            
            # Run inference
            with torch.no_grad():
                outputs = self.model(input_tensor, equation_coeffs=coeffs)
            
            # Create visualizations
            segmentation_overlay = create_segmentation_overlay(
                image.resize((self.config.data.image_size, self.config.data.image_size)),
                outputs['region_logits'],
                self.config.data.class_names,
                self.config.visualization.segmentation_colors
            )
            
            anomaly_heatmap = create_anomaly_heatmap(outputs['anomaly_map'])
            
            # Process results
            similarity_score = outputs['final_similarity_score'].item()
            predicted_class_idx = torch.argmax(outputs['region_logits']).item()
            predicted_class = self.config.data.class_names[predicted_class_idx]
            
            # Check if meets threshold
            meets_threshold = similarity_score >= self.config.similarity.threshold
            status = "PASS" if meets_threshold else "FAIL"
            status_color = "green" if meets_threshold else "red"
            
            # Create results text
            results_text = f"""
            **Analysis Complete**
            
            **Final Similarity Score:** {similarity_score:.4f}
            
            **Status:** <span style='color:{status_color}; font-weight:bold;'>{status}</span>
            
            **Predicted Primary Region:** {predicted_class.capitalize()}
            
            **Classification Confidence:** {outputs['classification_confidence'].item():.3f}
            
            **Anomaly Score:** {outputs['anomaly_score'].item():.3f}
            """
            
            # Class probabilities
            class_probs = F.softmax(outputs['region_logits'], dim=1).squeeze()
            class_probabilities = {
                name: prob.item() 
                for name, prob in zip(self.config.data.class_names, class_probs)
            }
            
            return {
                'segmentation': segmentation_overlay,
                'anomaly_heatmap': anomaly_heatmap,
                'results_text': results_text,
                'class_probabilities': class_probabilities,
                'similarity_score': similarity_score,
                'status': status,
                'predicted_class': predicted_class
            }
            
        except Exception as e:
            self.logger.error(f"Analysis failed: {e}")
            return {
                'error': str(e),
                'segmentation': None,
                'anomaly_heatmap': None,
                'results_text': f'Analysis failed: {str(e)}',
                'class_probabilities': {}
            }
    
    def _create_gradio_interface(self):
        """Create the Gradio web interface."""
        def analyze_wrapper(image, coeff_A, coeff_B, coeff_C):
            """Wrapper function for Gradio interface."""
            results = self.analyze_image(image, coeff_A, coeff_B, coeff_C)
            
            if 'error' in results:
                return (
                    None,  # segmentation
                    None,  # anomaly heatmap
                    results['results_text'],  # results text
                    {}     # class probabilities
                )
            
            return (
                results['segmentation'],
                results['anomaly_heatmap'],
                results['results_text'],
                results['class_probabilities']
            )
        
        # Create the interface
        with gr.Blocks(
            theme=gr.themes.Soft(),
            title="Fiber Optic Analysis",
            css=".markdown-body { font-family: 'Segoe UI', sans-serif; }"
        ) as interface:
            
            gr.Markdown("# 🔬 Fiber Optic Analysis System")
            gr.Markdown(
                "Upload a fiber optic image to analyze its condition and quality. "
                "Adjust the equation coefficients to see how they affect the final similarity score."
            )
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input section
                    gr.Markdown("### 📁 Upload Image")
                    input_image = gr.Image(
                        type="pil", 
                        label="Fiber Optic Image",
                        height=300
                    )
                    
                    # Coefficient controls
                    gr.Markdown("### ⚙️ Equation Coefficients")
                    gr.Markdown("*Adjust the weights for: S = A×(classification) + B×(1-anomaly) + C×(similarity)*")
                    
                    coeff_A = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=self.config.equation.coefficients.A,
                        step=0.1,
                        label="A - Classification Confidence Weight"
                    )
                    
                    coeff_B = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=self.config.equation.coefficients.B,
                        step=0.1,
                        label="B - Anomaly Detection Weight"
                    )
                    
                    coeff_C = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=self.config.equation.coefficients.C,
                        step=0.1,
                        label="C - Embedding Similarity Weight"
                    )
                    
                    # Analyze button
                    analyze_btn = gr.Button(
                        "🔍 Analyze Image",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=2):
                    # Results section
                    gr.Markdown("### 📊 Analysis Results")
                    
                    with gr.Row():
                        segmentation_output = gr.Image(
                            label="Segmentation Overlay",
                            height=250
                        )
                        anomaly_output = gr.Image(
                            label="Anomaly Heatmap",
                            height=250
                        )
                    
                    results_text = gr.Markdown()
                    
                    class_probabilities = gr.Label(
                        label="Region Classification Probabilities",
                        num_top_classes=len(self.config.data.class_names)
                    )
            
            # Example images section
            gr.Markdown("### 📷 Example Images")
            gr.Markdown("Click on an example image to load it for analysis.")
            
            # You can add example images here
            # example_images = gr.Dataset(...)
            
            # Connect the interface
            inputs = [input_image, coeff_A, coeff_B, coeff_C]
            outputs = [segmentation_output, anomaly_output, results_text, class_probabilities]
            
            analyze_btn.click(
                fn=analyze_wrapper,
                inputs=inputs,
                outputs=outputs
            )
            
            # Auto-analyze when image is uploaded
            input_image.change(
                fn=analyze_wrapper,
                inputs=inputs,
                outputs=outputs
            )
        
        return interface
    
    def launch(self, share=None, server_name=None, server_port=None, **kwargs):
        """Launch the web interface using config settings."""
        if not GRADIO_AVAILABLE:
            self.logger.error("Gradio not available. Install with: pip install gradio")
            return
        
        if self.interface is None:
            self.logger.error("Interface not initialized")
            return
        
        # Use config values if not provided
        if share is None:
            share = self.config.webapp.share
        if server_name is None:
            server_name = self.config.webapp.host
        if server_port is None:
            server_port = self.config.webapp.port
        
        self.logger.info(f"Launching web interface on {server_name}:{server_port}")
        
        try:
            self.interface.launch(
                share=share,
                server_name=server_name,
                server_port=server_port,
                **kwargs
            )
        except Exception as e:
            self.logger.error(f"Failed to launch interface: {e}")

def main():
    """Main function for running the web app."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    # Create and launch app using config settings
    try:
        app = FiberOpticAnalysisApp()
        app.launch()
    except Exception as e:
        logging.error(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
