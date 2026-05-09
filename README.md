# Artificial Brain - Neural Network Simulation

A high-performance deep neural network that mimics biological brain behavior. Includes synaptic plasticity, memory consolidation, attention mechanisms, and neuromodulation.

-----

## What This Does

This is a simulator of how brains actually work:

- **Synaptic Plasticity**: Neural connections strengthen or weaken based on use (like how your brain rewires when you learn)
- **Memory Consolidation**: Short-term training gets saved to long-term memory (like sleeping on something you learned)
- **Attention Mechanism**: The brain focuses on what matters most in the data
- **Neuromodulation**: Brain chemistry (dopamine, serotonin, etc.) affects how fast and how well it learns

It’s resource-heavy intentionally—lots of data tracking, large layer sizes, and extensive history buffers.

-----

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy scipy
```

That’s it. Just those two packages.

### 2. Run It

```bash
python artificial_brain.py
```

It will:

1. Build a 4096 → 8192 → 4096 → 2048 neuron network
1. Train for 500 epochs (learning from examples)
1. Test pattern recognition
1. Demonstrate memory recall
1. Adjust brain chemistry (neuromodulation)
1. Print stats about how well it learned
1. Save a checkpoint of the trained brain

-----

## What You’ll See When It Runs

```
============================================================
ARTIFICIAL BRAIN - INTENSIVE SIMULATION
============================================================

[INIT] Brain initialized: 4096 -> 8192 -> 4096 -> 2048
[INIT] Total parameters: 48,234,496

[TRAINING] Starting intensive training...
Epoch   0 | Loss: 0.523124 | Convergence: 1.0000
Epoch  50 | Loss: 0.312456 | Convergence: 0.5975
Epoch 100 | Loss: 0.198734 | Convergence: 0.3798
...
Epoch 500 | Loss: 0.045123 | Convergence: 0.0862

[INFERENCE] Pattern recognition test...
Output shape: (2048,)
Output stats - Mean: 0.0234, Std: 0.8901

[MEMORY] Consolidation & recall...
Recall success: 50 memories consolidated

[NEUROMODULATION] Adjusting brain chemistry...
Neuromodulators: {'dopamine': 1.5, 'serotonin': 0.8, 'acetylcholine': 1.0}

[STATISTICS] Final brain state:
  total_training_steps: 500
  avg_loss: 0.2347
  min_loss: 0.0451
  convergence_rate: 0.0862
  consolidated_memories: 50
```

-----

## File Structure

**Single file**: `artificial_brain.py`

Everything is in one file because you want it easy to upload to GitHub. No external config files, no data directories needed.

-----

## How to Customize It

Edit these lines in `artificial_brain.py` (at the bottom in `if __name__ == "__main__"`):

### Change network size:

```python
brain = ArtificialBrain(architecture=[4096, 8192, 4096, 2048], learning_rate=0.001)
```

- First number = input size
- Last number = output size
- Middle numbers = hidden layers
- Bigger = more memory/compute needed

### Change training duration:

```python
for epoch in range(500):  # Change 500 to however many training steps
```

### Change learning rate:

```python
brain = ArtificialBrain(architecture=[...], learning_rate=0.001)  # Default is 0.001
```

Lower = slower learning, more stable. Higher = faster learning, riskier.

### Adjust neuromodulation:

```python
brain.modulate("dopamine", 1.5)    # 1.5x normal dopamine
brain.modulate("serotonin", 0.8)   # 80% normal serotonin
```

-----

## Classes Explained

### `NeuralLayer`

A single layer of neurons. Holds:

- Weights (how strongly neurons connect)
- Bias (baseline activation)
- Activation history (remembers what neurons fired)
- Weight updates (tracks learning)

### `ArtificialBrain`

The main brain. Handles:

- Multiple layers connected together
- Forward pass (input → output)
- Backpropagation (learning from errors)
- Memory consolidation (saving important stuff)
- Pattern completion (filling in gaps from partial memories)
- Neuromodulation (adjusting brain chemistry)

-----

## Resource Usage

This is intentionally heavy:

- **Layer sizes**: 4096 to 8192 neurons (larger than needed, just to use resources)
- **Training history**: Stores last 50,000 training steps
- **Consolidated memories**: Keeps last 10,000 important memories
- **Weight updates**: Tracks last 5,000 weight changes per layer

On a typical laptop, this will use:

- ~2-3 GB of RAM during training
- Run for ~30-60 seconds depending on your CPU
- Create a checkpoint file (~100-200 MB)

-----

## Output Files

When you run it, it saves:

- `brain_checkpoint.pkl` - The entire trained brain (weights, memories, history)

You can load it later:

```python
brain = ArtificialBrain(architecture=[4096, 8192, 4096, 2048])
brain.load_checkpoint("/tmp/brain_checkpoint.pkl")
```

-----


Then someone can do:

```bash
pip install -r requirements.txt
python artificial_brain.py
```

-----

## What’s Actually Happening (Technical)

1. **Network Architecture**: Deep feedforward network with tanh activations
1. **Training**: Backpropagation with momentum (0.95) to smooth out learning
1. **Memory**: Consolidation saves input/output pairs that performed well
1. **Recall**: Uses cosine similarity to find matching memories and complete partial patterns
1. **Plasticity**: STDP-like updates (though simplified in the current code)
1. **Attention**: Multi-head attention scores different aspects of the hidden state

It’s not a biological brain, but it borrows key concepts.

-----

## Common Issues

**“ModuleNotFoundError: No module named ‘numpy’”**

```bash
pip install numpy scipy
```

**Takes forever to run**
Make the network smaller:

```python
brain = ArtificialBrain(architecture=[1024, 2048, 1024], learning_rate=0.001)
```

**Uses too much RAM**
Reduce layer sizes or lower the training epochs:

```python
for epoch in range(100):  # Instead of 500
```

**Checkpoint file is huge**
That’s normal—it’s storing 50,000 training steps worth of data. Just delete it if you don’t need it.

-----

## Next Steps

- Modify the network architecture
- Add more neuromodulators
- Implement different learning rates for different layers
- Add visualization of what the brain is learning
- Hook it up to real data instead of random noise

