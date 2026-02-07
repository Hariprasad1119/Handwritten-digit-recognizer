"""
MNIST Handwritten Digit Recognizer - Training Script
Trains a CNN model to recognize handwritten digits (0-9)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime

def create_cnn_model(input_shape=(28, 28, 1), num_classes=10):
    """
    Create a Convolutional Neural Network for digit recognition
    
    Architecture:
    - 2 Convolutional layers with MaxPooling
    - Dropout for regularization
    - Dense layers for classification
    """
    model = keras.Sequential([
        # First Convolutional Block
        layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Second Convolutional Block
        layers.Conv2D(64, kernel_size=(3, 3), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        
        # Flatten and Dense layers
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def load_and_preprocess_data():
    """Load MNIST dataset and preprocess it"""
    print("Loading MNIST dataset...")
    
    # Load data
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Normalize pixel values to [0, 1]
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # Reshape to add channel dimension (required for CNN)
    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)
    
    print(f"Training samples: {x_train.shape[0]}")
    print(f"Test samples: {x_test.shape[0]}")
    print(f"Image shape: {x_train.shape[1:]}")
    
    return (x_train, y_train), (x_test, y_test)

def plot_training_history(history, save_path='results'):
    """Plot and save training history"""
    os.makedirs(save_path, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_history.png', dpi=300, bbox_inches='tight')
    print(f"Training history plot saved to {save_path}/training_history.png")
    plt.close()

def visualize_predictions(model, x_test, y_test, num_samples=10, save_path='results'):
    """Visualize model predictions on test samples"""
    os.makedirs(save_path, exist_ok=True)
    
    # Get random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    for i, idx in enumerate(indices):
        # Make prediction
        img = x_test[idx]
        prediction = model.predict(np.expand_dims(img, 0), verbose=0)
        predicted_label = np.argmax(prediction)
        true_label = y_test[idx]
        
        # Plot
        axes[i].imshow(img.squeeze(), cmap='gray')
        color = 'green' if predicted_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}', color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/predictions.png', dpi=300, bbox_inches='tight')
    print(f"Predictions visualization saved to {save_path}/predictions.png")
    plt.close()

def main():
    """Main training function"""
    print("="*60)
    print("MNIST Handwritten Digit Recognizer - CNN Training")
    print("="*60)
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # Create model
    print("\nCreating CNN model...")
    model = create_cnn_model()
    
    # Compile model
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Display model architecture
    print("\nModel Architecture:")
    model.summary()
    
    # Define callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-7
        )
    ]
    
    # Train model
    print("\nTraining model...")
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=20,
        validation_split=0.1,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\nEvaluating model on test set...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy*100:.2f}%")
    
    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f'models/mnist_cnn_{timestamp}.keras'
    model.save(model_path)
    print(f"\nModel saved to {model_path}")
    
    # Also save in h5 format for compatibility
    model_h5_path = f'models/mnist_cnn_model.h5'
    model.save(model_h5_path)
    print(f"Model saved to {model_h5_path}")
    
    # Plot training history
    print("\nGenerating training visualizations...")
    plot_training_history(history)
    
    # Visualize predictions
    visualize_predictions(model, x_test, y_test)
    
    # Save training summary
    with open('results/training_summary.txt', 'w') as f:
        f.write("MNIST Digit Recognizer - Training Summary\n")
        f.write("="*50 + "\n\n")
        f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Epochs: {len(history.history['loss'])}\n")
        f.write(f"Final Training Accuracy: {history.history['accuracy'][-1]*100:.2f}%\n")
        f.write(f"Final Validation Accuracy: {history.history['val_accuracy'][-1]*100:.2f}%\n")
        f.write(f"Test Accuracy: {test_accuracy*100:.2f}%\n")
        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"\nModel saved to: {model_path}\n")
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print(f"✓ Model saved to models/")
    print(f"✓ Results saved to results/")
    print("="*60)

if __name__ == "__main__":
    main()
