"""
Example Usage Script
Demonstrates how to use the MNIST digit recognizer programmatically
"""

import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def example_1_load_and_predict():
    """Example 1: Load trained model and make predictions"""
    print("\n" + "="*60)
    print("Example 1: Loading Model and Making Predictions")
    print("="*60)
    
    # Load the trained model
    model = keras.models.load_model('models/mnist_cnn_model.h5')
    print("✓ Model loaded successfully")
    
    # Load MNIST test data
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    
    # Preprocess
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)
    
    # Make predictions on first 5 samples
    sample_images = x_test[:5]
    predictions = model.predict(sample_images, verbose=0)
    
    print("\nPredictions for first 5 test samples:")
    for i, pred in enumerate(predictions):
        predicted_digit = np.argmax(pred)
        confidence = np.max(pred) * 100
        true_digit = y_test[i]
        
        print(f"  Sample {i+1}: Predicted={predicted_digit}, "
              f"True={true_digit}, Confidence={confidence:.2f}%")

def example_2_evaluate_performance():
    """Example 2: Evaluate model performance"""
    print("\n" + "="*60)
    print("Example 2: Model Performance Evaluation")
    print("="*60)
    
    # Load model
    model = keras.models.load_model('models/mnist_cnn_model.h5')
    
    # Load test data
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test.astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)
    
    # Evaluate
    print("\nEvaluating on full test set (10,000 samples)...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    
    print(f"\n✓ Test Loss: {test_loss:.4f}")
    print(f"✓ Test Accuracy: {test_accuracy*100:.2f}%")
    print(f"✓ Error Rate: {(1-test_accuracy)*100:.2f}%")

def example_3_confusion_analysis():
    """Example 3: Analyze which digits are most confused"""
    print("\n" + "="*60)
    print("Example 3: Confusion Analysis")
    print("="*60)
    
    # Load model
    model = keras.models.load_model('models/mnist_cnn_model.h5')
    
    # Load test data (use subset for speed)
    (_, _), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = x_test[:1000].astype('float32') / 255.0
    x_test = np.expand_dims(x_test, -1)
    y_test = y_test[:1000]
    
    # Get predictions
    predictions = model.predict(x_test, verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)
    
    # Find misclassifications
    misclassified = predicted_labels != y_test
    num_misclassified = np.sum(misclassified)
    
    print(f"\n✓ Total samples analyzed: {len(y_test)}")
    print(f"✓ Misclassified: {num_misclassified}")
    print(f"✓ Accuracy: {(1 - num_misclassified/len(y_test))*100:.2f}%")
    
    if num_misclassified > 0:
        print("\nMost common misclassifications:")
        misclass_pairs = list(zip(y_test[misclassified], 
                                 predicted_labels[misclassified]))
        from collections import Counter
        most_common = Counter(misclass_pairs).most_common(5)
        
        for (true, pred), count in most_common:
            print(f"  {true} → {pred}: {count} times")

def example_4_custom_prediction():
    """Example 4: Create and predict custom digit"""
    print("\n" + "="*60)
    print("Example 4: Custom Digit Creation and Prediction")
    print("="*60)
    
    # Create a simple custom digit (number 7)
    custom_digit = np.zeros((28, 28))
    # Draw a 7
    custom_digit[5:8, 5:23] = 1.0    # Top horizontal line
    custom_digit[8:11, 20:23] = 1.0  # Top right corner
    custom_digit[11:14, 17:20] = 1.0 # Diagonal
    custom_digit[14:17, 14:17] = 1.0
    custom_digit[17:20, 11:14] = 1.0
    custom_digit[20:25, 8:11] = 1.0  # Bottom vertical
    
    # Reshape for model
    custom_digit_input = np.expand_dims(custom_digit, axis=(0, -1))
    
    # Load model and predict
    model = keras.models.load_model('models/mnist_cnn_model.h5')
    prediction = model.predict(custom_digit_input, verbose=0)
    predicted_digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    
    print(f"\n✓ Created custom digit")
    print(f"✓ Predicted: {predicted_digit}")
    print(f"✓ Confidence: {confidence:.2f}%")
    
    # Visualize
    plt.figure(figsize=(6, 6))
    plt.imshow(custom_digit, cmap='gray')
    plt.title(f'Custom Digit\nPrediction: {predicted_digit} ({confidence:.1f}%)')
    plt.axis('off')
    plt.savefig('results/custom_digit_example.png', dpi=150, bbox_inches='tight')
    print("✓ Visualization saved to results/custom_digit_example.png")
    plt.close()

def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" "*15 + "MNIST DIGIT RECOGNIZER - EXAMPLES")
    print("="*70)
    
    try:
        example_1_load_and_predict()
        example_2_evaluate_performance()
        example_3_confusion_analysis()
        example_4_custom_prediction()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        
    except FileNotFoundError:
        print("\n" + "="*70)
        print("ERROR: Model file not found!")
        print("Please train the model first by running: python train_model.py")
        print("="*70)
    
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure you have trained the model first.")

if __name__ == "__main__":
    main()
