from ultralytics import YOLO

if __name__ == "__main__":
    # Load a pretrained YOLO model
    model = YOLO("yolo11n.pt")  # Make sure this weights file exists in your directory

    # Train the model
    model.train(
        data="data/data/data.yaml",   # Path to your dataset YAML
        epochs=200,                   # Number of training epochs
        batch=4,                      # Batch size
        imgsz=1200,                   # Image size
        device=0,                     # GPU device ID (0 = first GPU), set to "cpu" if needed
        patience=20,                  # Early stopping patience
        name="project_3_model"        # Name for saving this training run
    )
