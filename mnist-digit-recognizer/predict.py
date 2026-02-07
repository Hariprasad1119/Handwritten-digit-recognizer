"""
MNIST Digit Recognizer - Prediction Script
Use trained model to predict handwritten digits
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import os

def load_model(model_path='models/mnist_cnn_model.h5'):
    """Load the trained model"""
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first by running: python train_model.py")
        sys.exit(1)
    
    print(f"Loading model from {model_path}...")
    model = keras.models.load_model(model_path)
    print("Model loaded successfully!")
    return model

def preprocess_image(image_path):
    """
    Preprocess an image for prediction
    Converts to grayscale, resizes to 28x28, normalizes
    """
    try:
        # Load image
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        
        # Resize to 28x28
        img = img.resize((28, 28), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(img)
        
        # Invert if needed (MNIST has white digits on black background)
        # Check if background is white
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
        
        # Normalize to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # Add batch and channel dimensions
        img_array = np.expand_dims(img_array, axis=(0, -1))
        
        return img_array
    
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

def predict_digit(model, image_path, show_plot=True):
    """Predict digit from an image file"""
    # Preprocess image
    img_array = preprocess_image(image_path)
    
    if img_array is None:
        return None
    
    # Make prediction
    prediction = model.predict(img_array, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    # Display results
    print(f"\nPrediction for {image_path}:")
    print(f"Predicted Digit: {predicted_digit}")
    print(f"Confidence: {confidence:.2f}%")
    print(f"\nAll class probabilities:")
    for digit, prob in enumerate(prediction[0]):
        print(f"  Digit {digit}: {prob*100:.2f}%")
    
    # Visualize if requested
    if show_plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        
        # Show original image
        original_img = Image.open(image_path).convert('L')
        ax1.imshow(original_img, cmap='gray')
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # Show probability distribution
        ax2.bar(range(10), prediction[0])
        ax2.set_xlabel('Digit')
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Prediction: {predicted_digit} ({confidence:.1f}%)')
        ax2.set_xticks(range(10))
        
        plt.tight_layout()
        plt.savefig('results/latest_prediction.png', dpi=150, bbox_inches='tight')
        plt.show()
        print("\nVisualization saved to results/latest_prediction.png")
    
    return predicted_digit, confidence

def predict_from_mnist_test():
    """Load test samples from MNIST and predict"""
    print("\nLoading MNIST test samples...")
    
    # Load MNIST test data
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)
    
    # Load model
    model = load_model()
    
    # Get random samples
    num_samples = 5
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(15, 3))
    
    print("\nPredicting on random MNIST test samples:\n")
    
    for i, idx in enumerate(indices):
        img = x_test[idx]
        true_label = y_test[idx]
        
        # Predict
        prediction = model.predict(np.expand_dims(img, 0), verbose=0)
        predicted_label = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        
        # Display
        axes[i].imshow(img.squeeze(), cmap='gray')
        color = 'green' if predicted_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}\n({confidence:.1f}%)', 
                         color=color, fontsize=10)
        axes[i].axis('off')
        
        print(f"Sample {i+1}: True={true_label}, Predicted={predicted_label}, Confidence={confidence:.2f}%")
    
    plt.tight_layout()
    plt.savefig('results/mnist_test_predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("\nVisualization saved to results/mnist_test_predictions.png")

def interactive_mode():
    """Interactive prediction mode"""
    model = load_model()
    
    print("\n" + "="*60)
    print("Interactive Digit Prediction Mode")
    print("="*60)
    print("\nOptions:")
    print("1. Predict from your own image file")
    print("2. Test on random MNIST samples")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            image_path = input("Enter path to image file: ").strip()
            if os.path.exists(image_path):
                predict_digit(model, image_path)
            else:
                print(f"Error: File not found: {image_path}")
        
        elif choice == '2':
            predict_from_mnist_test()
        
        elif choice == '3':
            print("Exiting...")
            break
        
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")

def main():
    """Main function"""
    os.makedirs('results', exist_ok=True)
    
    if len(sys.argv) > 1:
        # Command line mode - predict specific image
        image_path = sys.argv[1]
        if os.path.exists(image_path):
            model = load_model()
            predict_digit(model, image_path)
        else:
            print(f"Error: Image file not found: {image_path}")
    else:
        # Interactive mode
        interactive_mode()

if __name__ == "__main__":
    main()
