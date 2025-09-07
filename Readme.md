# 🛍️ Product Matching with Siamese Networks

A high-performance deep learning solution for e-commerce product matching using Siamese Neural Networks with pre-computed SBERT embeddings.

![Python](\download.png)

## 🎯 Overview

This project implements a Siamese Neural Network architecture for identifying matching product pairs in e-commerce catalogs. By leveraging pre-computed Sentence-BERT embeddings and a trainable projection head, the system achieves exceptional accuracy while maintaining computational efficiency.

### Key Features

- 🚀 **High Performance**: 96.61% precision, 99.70% recall, 98.13% F1-score
- ⚡ **Efficient Training**: Pre-computed SBERT embeddings reduce training time significantly
- 🎯 **Lightweight Architecture**: Only projection head requires training
- 📊 **Robust Evaluation**: Comprehensive metrics and distance analysis
- 🔄 **Easy Inference**: Simple API for real-time product matching

## 🏗️ Architecture

```
Product Title 1 → SBERT Encoder → Projection Head ┐
                                                   ├→ Contrastive Loss
Product Title 2 → SBERT Encoder → Projection Head ┘
```

### Components

1. **Pre-computed SBERT Embeddings**: 384-dimensional vectors from `all-MiniLM-L6-v2`
2. **Siamese Network**: Twin projection heads with shared weights
3. **Contrastive Loss**: Distance-based loss function for similarity learning

## 📋 Requirements

```txt
torch>=1.9.0
pandas>=1.3.0
sentence-transformers>=2.0.0
scikit-learn>=1.0.0
numpy>=1.21.0
pickle
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/product-matching-siamese.git
cd product-matching-siamese

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import pandas as pd
from siamese_matcher import train_model, SiameseNetwork, load_and_prepare_data

# Load and prepare data
train_loader, test_loader, embedding_map = load_and_prepare_data(
    'path/to/your/training_pairs.csv',
    batch_size=32,
    test_split=0.2
)

# Initialize and train model
model = SiameseNetwork(input_dim=384, embedding_dim=128)
trained_model = train_model(
    model, 
    train_loader, 
    num_epochs=20, 
    learning_rate=0.001
)

# Evaluate model
precision, recall, f1, predictions, labels, distances = evaluate_model(
    trained_model, 
    test_loader, 
    threshold=0.6
)
```

## 📁 Project Structure

```
product-matching-siamese/
│
├── data/
│   └── training_pairs.csv          # Training dataset
│
├── src/
│   ├── dataset.py                  # ProductPairDataset class
│   ├── model.py                    # SiameseNetwork architecture
│   ├── loss.py                     # ContrastiveLoss implementation
│   ├── train.py                    # Training utilities
│   ├── evaluate.py                 # Evaluation functions
│   └── inference.py                # Inference pipeline
│
├── notebooks/
│   └── product_matching_demo.ipynb # Jupyter notebook demo
│
├── models/
│   └── siamese_matcher.pth         # Saved model weights
│
├── requirements.txt
├── README.md
└── setup.py
```

## 📊 Dataset Format

Your CSV file should contain the following columns:

| title1 | title2 | match |
|--------|--------|-------|
| iPhone 13 Pro Max 256GB Blue | Apple iPhone 13 Pro Max 256GB Blue | 1 |
| Samsung Galaxy S21 | iPhone 12 Pro | 0 |
| Nike Air Force 1 White | Nike Air Force One White Sneakers | 1 |

- `title1`, `title2`: Product titles to compare
- `match`: Binary label (1 for matching products, 0 for non-matching)

## 🎯 Performance Metrics

| Metric | Score |
|--------|-------|
| **Precision** | 96.61% |
| **Recall** | 99.70% |
| **F1-Score** | 98.13% |
| **Avg Distance (Matches)** | 0.1225 |
| **Avg Distance (Non-matches)** | 1.3318 |

## 🔧 Configuration

### Hyperparameters

```python
# Model Configuration
INPUT_DIM = 384          # SBERT embedding dimension
EMBEDDING_DIM = 128      # Output embedding dimension

# Training Configuration
BATCH_SIZE = 32
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
MARGIN = 1.0            # Contrastive loss margin
THRESHOLD = 0.6         # Classification threshold
```

### Advanced Configuration

```python
# Custom model with different architecture
model = SiameseNetwork(
    input_dim=384,
    embedding_dim=256,      # Larger embedding space
    hidden_dim=512         # Custom hidden layer size
)

# Custom training with different parameters
trained_model = train_model(
    model,
    train_loader,
    num_epochs=50,
    learning_rate=0.0005,
    weight_decay=1e-4,
    scheduler='cosine'      # Learning rate scheduling
)
```

## 🚀 Inference Examples

### Single Pair Matching

```python
from inference import ProductMatcher

matcher = ProductMatcher('models/siamese_matcher.pth')

# Check if two products match
result = matcher.match_products(
    "iPhone 13 Pro Max 256GB Blue",
    "Apple iPhone 13 Pro Max 256GB Blue"
)

print(f"Distance: {result['distance']:.4f}")
print(f"Match: {result['is_match']}")
# Output: Distance: 0.0373, Match: True
```

### Batch Processing

```python
# Match multiple product pairs
pairs = [
    ("iPhone 13 Pro Max 256GB Blue", "Apple iPhone 13 Pro Max 256GB Blue"),
    ("Samsung Galaxy S21", "iPhone 12 Pro"),
    ("Nike Air Force 1 White", "Nike Air Force One White Sneakers")
]

results = matcher.match_batch(pairs)
for result in results:
    print(f"{result['pair']}: {result['is_match']} (dist: {result['distance']:.4f})")
```

## 📈 Training Tips

### Data Preparation
- Ensure balanced dataset (equal matching and non-matching pairs)
- Clean product titles (remove special characters, normalize spacing)
- Consider data augmentation for small datasets

### Model Optimization
- Start with lower learning rates (0.0001) for fine-tuning
- Use learning rate scheduling for better convergence
- Monitor validation loss to prevent overfitting
- Experiment with different margin values (0.5-2.0)

### Performance Tuning
- Adjust threshold based on precision/recall requirements
- Use cross-validation for robust evaluation
- Consider ensemble methods for production deployment

## 🛠️ Advanced Features

### Custom Loss Functions

```python
class WeightedContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0, pos_weight=1.0):
        super().__init__()
        self.margin = margin
        self.pos_weight = pos_weight
    
    def forward(self, embedding1, embedding2, label):
        # Custom weighted loss implementation
        pass
```

### Multi-GPU Training

```python
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
    
# Training with multiple GPUs
trained_model = train_model(
    model,
    train_loader,
    device='cuda',
    multi_gpu=True
)
```

### Model Deployment

```python
# Export model for production
torch.jit.script(model).save('siamese_matcher_scripted.pt')

# Load scripted model (faster inference)
scripted_model = torch.jit.load('siamese_matcher_scripted.pt')
```

## 📊 Monitoring and Logging

### Training Monitoring

```python
import wandb

# Initialize Weights & Biases
wandb.init(project="product-matching")

# Log metrics during training
wandb.log({
    "epoch": epoch,
    "train_loss": avg_loss,
    "learning_rate": optimizer.param_groups[0]['lr']
})
```

### Performance Analysis

```python
# Analyze prediction confidence
distances = evaluate_model(model, test_loader, return_distances=True)
# Analyze distribution of distances for matches vs non-matches
match_distances = distances[labels == 1]  
non_match_distances = distances[labels == 0]
print(f"Match distances - Mean: {match_distances.mean():.4f}, Std: {match_distances.std():.4f}")
print(f"Non-match distances - Mean: {non_match_distances.mean():.4f}, Std: {non_match_distances.std():.4f}")
```

## 🧪 Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

Run specific tests:

```bash
python -m pytest tests/test_model.py::test_siamese_forward -v
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run linting
flake8 src/
black src/

# Run type checking
mypy src/
```

## 📚 References

- [Siamese Neural Networks for One-shot Image Recognition](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf)
- [Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks](https://arxiv.org/abs/1908.10084)
- [Learning a Similarity Metric Discriminatively](http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Sentence-Transformers library for pre-trained embeddings
- PyTorch team for the deep learning framework
- Contributors and researchers in the product matching domain

## 📞 Support

- 📧 Email: your.email@example.com
- 🐛 Issues: [GitHub Issues](https://github.com/yourusername/product-matching-siamese/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/yourusername/product-matching-siamese/discussions)

---

⭐ **Star this repository if you found it helpful!**

Made with ❤️ for the e-commerce ML community