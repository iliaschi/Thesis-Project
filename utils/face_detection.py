import numpy as np
from facenet_pytorch import MTCNN
import torch
from PIL import Image, ImageEnhance
import os
import time
from tqdm import tqdm
from datetime import datetime
import csv
import logging

class FaceDetector:
    def __init__(self, min_face_size=100, confidence_threshold=0.9, device='cpu'):
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('FaceDetector')
        
        # Initialize detector settings
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.mtcnn = MTCNN(
            min_face_size=min_face_size,
            thresholds=[0.6, 0.7, 0.7],
            device=device,
            keep_all=True,
            selection_method='probability'
        )
        
        # Tracking statistics
        self.stats = {
            'total_images': 0,
            'total_faces': 0,
            'high_confidence_faces': 0,
            'low_confidence_faces': 0,
            'errors': 0,
            'processing_time': 0
        }

    def enhance_image(self, img):
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        return img

    def detect_faces(self, image_path, save_dir=None, emotion=None, enhance=True):
        try:
            # Load and optionally enhance image
            self.logger.info(f"Loading image: {image_path}")
            img = Image.open(image_path)
            if enhance:
                img = self.enhance_image(img)
            
            # Detect faces
            boxes, probs = self.mtcnn.detect(img)
            
            if boxes is None:
                self.logger.info(f"No faces detected in {image_path}")
                return []
            
            faces = []
            high_conf_count = 0
            low_conf_count = 0
            
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                # Log confidence for debugging
                self.logger.info(f"Face {i} confidence: {prob:.3f}")
                
                # Check confidence threshold
                if prob < self.confidence_threshold:
                    low_conf_count += 1
                    continue
                
                high_conf_count += 1
                
                # Get coordinates with padding
                x1, y1, x2, y2 = box.astype(int)
                
                # Add padding
                h, w = img.height, img.width
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.05)
                
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                
                # Crop face
                face = img.crop((x1, y1, x2, y2))
                
                # Standardize size
                face = face.resize((256, 256), Image.LANCZOS)
                
                faces.append((face, prob))
                
                # Save if directory provided
                if save_dir and emotion:
                    os.makedirs(save_dir, exist_ok=True)
                    save_path = os.path.join(
                        save_dir, 
                        f"{os.path.basename(image_path).split('.')[0]}_face_{i}_conf_{prob:.2f}.jpg"
                    )
                    face.save(save_path, quality=95)
                    self.logger.info(f"Saved face to: {save_path}")
            
            # Update statistics
            self.stats['high_confidence_faces'] += high_conf_count
            self.stats['low_confidence_faces'] += low_conf_count
            
            return faces
            
        except Exception as e:
            self.stats['errors'] += 1
            self.logger.error(f"Error processing {image_path}: {str(e)}")
            return []

    def process_with_report(self, input_dir, output_dir, report_dir=None):
        start_time = time.time()
        self.stats = {
            'total_images': 0,
            'total_faces': 0,
            'high_confidence_faces': 0,
            'low_confidence_faces': 0,
            'errors': 0,
            'processing_time': 0,
            'per_emotion': {}
        }
        
        # Setup report directory
        if report_dir is None:
            report_dir = output_dir
        os.makedirs(report_dir, exist_ok=True)
        
        # CSV for reporting
        csv_path = os.path.join(report_dir, f"face_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
        
        # Check if input_dir exists
        if not os.path.exists(input_dir):
            self.logger.error(f"Input directory does not exist: {input_dir}")
            return self.stats
            
        # List all files directly (no emotion subdirectories)
        if os.path.isdir(input_dir):
            # Direct file listing
            self.logger.info(f"Scanning directory: {input_dir}")
            all_files = os.listdir(input_dir)
            self.logger.info(f"Found {len(all_files)} total files in directory")
            
            # Filter image files
            supported_extensions = ('.jpg', '.jpeg', '.png', '.jfif', '.jpe', '.gif', '.bmp')
            image_files = [f for f in all_files if f.lower().endswith(supported_extensions)]
            self.logger.info(f"Found {len(image_files)} image files")
            
            if not image_files:
                self.logger.warning(f"No image files found in {input_dir}")
                self.logger.info(f"Supported extensions: {supported_extensions}")
                return self.stats
            
            # Extract emotion from directory name if possible
            emotion = os.path.basename(input_dir)
            
            # Process images
            with open(csv_path, 'w', newline='') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerow(['Image', 'Face ID', 'Confidence', 'Status'])
                
                self.stats['total_images'] = len(image_files)
                
                for img_name in tqdm(image_files, desc=f"Processing images"):
                    image_path = os.path.join(input_dir, img_name)
                    
                    # Detect and save faces
                    faces = self.detect_faces(image_path, output_dir, emotion)
                    
                    # Log to CSV
                    if faces:
                        for i, (_, conf) in enumerate(faces):
                            csv_writer.writerow([img_name, i, f"{conf:.4f}", "Saved"])
                    else:
                        csv_writer.writerow([img_name, "None", "0.0000", "No faces/Low confidence"])
                    
                    # Update total stats
                    self.stats['total_faces'] += len(faces)
        
        # Calculate processing time
        self.stats['processing_time'] = time.time() - start_time
        
        # Generate summary report
        self._generate_summary_report(report_dir)
        
        return self.stats
    
    def _generate_summary_report(self, report_dir):
        """Generate summary report"""
        report_path = os.path.join(report_dir, f"summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        
        with open(report_path, 'w') as f:
            f.write("=================================================\n")
            f.write("           FACE DETECTION SUMMARY REPORT         \n")
            f.write("=================================================\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing Time: {self.stats['processing_time']:.2f} seconds\n\n")
            
            f.write("OVERALL STATISTICS:\n")
            f.write("-------------------------------------------------\n")
            f.write(f"Total Images Processed: {self.stats['total_images']}\n")
            f.write(f"Total Faces Detected: {self.stats['high_confidence_faces']}\n")
            total_detected = max(1, self.stats['high_confidence_faces'] + self.stats['low_confidence_faces'])
            f.write(f"High Confidence Faces: {self.stats['high_confidence_faces']} " +
                   f"({self.stats['high_confidence_faces']/total_detected*100:.1f}%)\n")
            f.write(f"Low Confidence Faces: {self.stats['low_confidence_faces']} " +
                   f"({self.stats['low_confidence_faces']/total_detected*100:.1f}%)\n")
            f.write(f"Errors Encountered: {self.stats['errors']}\n\n")
            
            f.write("=================================================\n")
            f.write("                 END OF REPORT                   \n")
            f.write("=================================================\n")
        
        self.logger.info(f"Summary report saved to: {report_path}")

def main():
    # Define paths
    input_dir = r'C:\Users\ilias\Python\Thesis-Project\data\synthetic\raw_grids\men\surprised_men'
    output_dir = r"C:\Users\ilias\Python\Thesis-Project\data\synthetic\processed_python\men\surprised_men_proc"
    
    # Create detector
    detector = FaceDetector(
        min_face_size=80,
        confidence_threshold=0.8,  # Slightly lower threshold for testing
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Process all images
    stats = detector.process_with_report(input_dir, output_dir)
    
    # Print summary
    print(f"\nProcessing Complete!")
    print(f"Total Images: {stats['total_images']}")
    print(f"Total Faces: {stats['high_confidence_faces'] + stats['low_confidence_faces']}") 
    print(f"High Confidence Faces: {stats['high_confidence_faces']}")
    print(f"Processing Time: {stats['processing_time']:.2f} seconds")

if __name__ == "__main__":
    main()


