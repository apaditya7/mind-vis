from shape_predictor import train_shape_predictor
import os

# Create models directory
os.makedirs('models', exist_ok=True)

# Train shape predictor for GOD dataset
train_shape_predictor(
    dataset_name='GOD',
    max_samples=100,  # Quick test
    num_epochs=20,    # Quick training
    save_path='models/shape_predictor_god.pth'
)

print("Training completed! Ready for GOD evaluation.")