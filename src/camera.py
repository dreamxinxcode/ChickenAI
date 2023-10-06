import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from PIL import Image, ImageDraw
import numpy as np

class Camera:
    def __init__(self) -> None:
        self.cap = cv2.VideoCapture(0)  # 0 for the default camera
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Set the width
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Set the height
        self.cap.set(cv2.CAP_PROP_FPS, 1)  # Limit the frame rate to 10 FPS
        self.transform = transforms.ToTensor()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load a pre-trained Faster R-CNN model
        self.model = fasterrcnn_resnet50_fpn(pretrained=True)
        self.model.to(self.device)
        self.model.eval()


    def start_capture(self) -> None:
        print('Streaming...')
        while True:
            ret, frame = self.cap.read()
            
            # Convert the frame to a PyTorch tensor
            img_tensor = self.transform(frame).to(self.device)
            
            # Perform object detection on the frame
            img_with_boxes = self.detect_chickens(img_tensor, frame)
            
            # Resize the processed frame to match the original frame dimensions
            img_with_boxes_resized = cv2.resize(img_with_boxes, (frame.shape[1], frame.shape[0]))
            
            # Display the processed frame
            cv2.imshow("ChickenAI", img_with_boxes_resized)
            
            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('Stopping stream...')
                break

        # Release the camera and close OpenCV windows
        self.cap.release()
        cv2.destroyAllWindows()

    def detect_chickens(self, img_tensor, frame) -> np.ndarray:
        with torch.no_grad():
            # Perform object detection using the pre-trained model
            predictions = self.model([img_tensor])
            
            # Get the predicted boxes and labels
            boxes = predictions[0]['boxes']
            labels = predictions[0]['labels']
            
            # Draw boxes around chickens and label them
            img_pil = transforms.ToPILImage()(img_tensor.cpu())
            draw = ImageDraw.Draw(img_pil)
            
            for i, label in enumerate(labels):
                if label == 1:  # Assuming class 1 corresponds to chickens
                    box = boxes[i].cpu().numpy().astype(int)
                    # Draw a rectangle around the chicken
                    draw.rectangle(box.tolist(), outline=(0, 255, 0), width=2)
                    # Label the chicken as "Chicken"
                    label_position = (box[0], box[1] - 10)
                    draw.text(label_position, "Chicken", fill=(0, 255, 0))
            
            # Convert the PIL image back to a NumPy array
            img_with_boxes = np.array(img_pil)
            
            return img_with_boxes