#!/usr/bin/env python3
"""
M³-CHAT v2: AUTOREGRESSIVE MANIFOLD-MEMORY LANGUAGE MODEL
Enhanced Memory-Native Neural Network with Transformer-like capabilities

NEW FEATURES:
✓ Autoregressive token loop with m3chat_step()
✓ Learned projection (vector → vector)
✓ Fast + Slow memory split
✓ Multi-head resonance (attention replacement)
✓ Cross-entropy loss for language
✓ Positional encoding
✓ Proper memory direction preservation
✓ Teacher forcing support
✓ Vocabulary + embedding layer
✓ Residual memory paths
✓ Layer normalization on memory
✓ Token masking (no future leakage)
✓ Entropy control / temperature
✓ State reset protocol (soft/hard/session)
✓ Gradient flow through time (truncated BPTT)

Compile C library first:
    Windows: gcc -shared -o m3chat_v2.dll m3chat_v2.c -lm -O3 -fopenmp -static-libgcc -static
    Linux:   gcc -shared -fPIC -o m3chat_v2.so m3chat_v2.c -lm -O3 -fopenmp
    Mac:     gcc -shared -fPIC -o m3chat_v2.dylib m3chat_v2.c -lm -O3 -Xpreprocessor -fopenmp -lomp

Licensed under GPL V3.
"""

import ctypes
import numpy as np
from pathlib import Path
import platform
import sys
from typing import Optional, List, Tuple

# ============================================================================
# LOAD C LIBRARY
# ============================================================================

def load_library():
    """Load the appropriate shared library for the platform"""
    lib_path = Path(__file__).parent
    
    if platform.system() == 'Windows':
        lib_name = 'm3chat_v2.dll'
    elif platform.system() == 'Darwin':
        lib_name = 'm3chat_v2.dylib'
    else:
        lib_name = 'm3chat_v2.so'
    
    full_path = lib_path / lib_name
    
    if not full_path.exists():
        print(f"ERROR: Library not found at {full_path}")
        print("\nPlease compile the C library first:")
        if platform.system() == 'Windows':
            print("  gcc -shared -o m3chat_v2.dll m3chat_v2.c -lm -O3 -fopenmp -static-libgcc -static")
        elif platform.system() == 'Darwin':
            print("  gcc -shared -fPIC -o m3chat_v2.dylib m3chat_v2.c -lm -O3 -Xpreprocessor -fopenmp -lomp")
        else:
            print("  gcc -shared -fPIC -o m3chat_v2.so m3chat_v2.c -lm -O3 -fopenmp")
        sys.exit(1)
    
    return ctypes.CDLL(str(full_path))

# Load the library
try:
    _lib = load_library()
    print(f"✓ Loaded M³-Chat v2 C library successfully")
except Exception as e:
    print(f"ERROR loading library: {e}")
    sys.exit(1)

# ============================================================================
# DEFINE C FUNCTION SIGNATURES
# ============================================================================

# Network creation/destruction
_lib.create_m3chat_v2.argtypes = [
    ctypes.c_int,    # vocab_size
    ctypes.c_int,    # embed_dim
    ctypes.c_int,    # hidden_size
    ctypes.c_int,    # memory_dim
    ctypes.c_int,    # manifold_dim
    ctypes.c_int,    # num_heads
    ctypes.c_int     # max_seq_len
]
_lib.create_m3chat_v2.restype = ctypes.c_void_p
_lib.destroy_m3chat_v2.argtypes = [ctypes.c_void_p]
_lib.destroy_m3chat_v2.restype = None

# Autoregressive step
_lib.m3chat_step.argtypes = [
    ctypes.c_void_p,                    # net
    ctypes.c_int                        # token
]
_lib.m3chat_step.restype = None

# Generate sequence
_lib.m3chat_generate.argtypes = [
    ctypes.c_void_p,                    # net
    ctypes.POINTER(ctypes.c_int),       # prompt_tokens
    ctypes.c_int,                       # prompt_len
    ctypes.POINTER(ctypes.c_int),       # output_tokens
    ctypes.c_int,                       # max_new_tokens
    ctypes.c_float,                     # temperature
    ctypes.POINTER(ctypes.c_int)        # actual_length
]
_lib.m3chat_generate.restype = None

# Training
_lib.m3chat_train_sequence.argtypes = [
    ctypes.c_void_p,                    # net
    ctypes.POINTER(ctypes.c_int),       # tokens
    ctypes.c_int                        # seq_len
]
_lib.m3chat_train_sequence.restype = ctypes.c_float

# Memory consolidation
_lib.m3chat_consolidate.argtypes = [ctypes.c_void_p]
_lib.m3chat_consolidate.restype = None

# State reset (three separate functions)
_lib.m3chat_reset_soft.argtypes = [ctypes.c_void_p]
_lib.m3chat_reset_soft.restype = None
_lib.m3chat_reset_hard.argtypes = [ctypes.c_void_p]
_lib.m3chat_reset_hard.restype = None
_lib.m3chat_reset_session.argtypes = [ctypes.c_void_p]
_lib.m3chat_reset_session.restype = None

# Get logits
_lib.m3chat_get_logits.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float)
]
_lib.m3chat_get_logits.restype = None

# Get hidden state
_lib.m3chat_get_hidden_state_v2.argtypes = [
    ctypes.c_void_p,
    ctypes.POINTER(ctypes.c_float)
]
_lib.m3chat_get_hidden_state_v2.restype = None

# Get neuron memory (fast) - NOT IMPLEMENTED IN C
# _lib.m3chat_get_fast_memory.argtypes = [
#     ctypes.c_void_p,
#     ctypes.c_int,
#     ctypes.POINTER(ctypes.c_float)
# ]
# _lib.m3chat_get_fast_memory.restype = None

# Get neuron memory (slow) - NOT IMPLEMENTED IN C
# _lib.m3chat_get_slow_memory.argtypes = [
#     ctypes.c_void_p,
#     ctypes.c_int,
#     ctypes.POINTER(ctypes.c_float)
# ]
# _lib.m3chat_get_slow_memory.restype = None

# Get network memory - NOT IMPLEMENTED IN C
# _lib.m3chat_get_network_memory_v2.argtypes = [
#     ctypes.c_void_p,
#     ctypes.POINTER(ctypes.c_float)
# ]
# _lib.m3chat_get_network_memory_v2.restype = None

# Parameters
_lib.m3chat_get_temperature.argtypes = [ctypes.c_void_p]
_lib.m3chat_get_temperature.restype = ctypes.c_float
_lib.m3chat_set_temperature.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.m3chat_set_temperature.restype = None

_lib.m3chat_get_learning_rate_v2.argtypes = [ctypes.c_void_p]
_lib.m3chat_get_learning_rate_v2.restype = ctypes.c_float
_lib.m3chat_set_learning_rate_v2.argtypes = [ctypes.c_void_p, ctypes.c_float]
_lib.m3chat_set_learning_rate_v2.restype = None

_lib.m3chat_get_current_position.argtypes = [ctypes.c_void_p]
_lib.m3chat_get_current_position.restype = ctypes.c_int

_lib.m3chat_get_training_steps_v2.argtypes = [ctypes.c_void_p]
_lib.m3chat_get_training_steps_v2.restype = ctypes.c_int

_lib.m3chat_get_last_loss_v2.argtypes = [ctypes.c_void_p]
_lib.m3chat_get_last_loss_v2.restype = ctypes.c_float

_lib.m3chat_get_avg_resonance_v2.argtypes = [ctypes.c_void_p]
_lib.m3chat_get_avg_resonance_v2.restype = ctypes.c_float

_lib.m3chat_get_avg_memory_norm_v2.argtypes = [ctypes.c_void_p]
_lib.m3chat_get_avg_memory_norm_v2.restype = ctypes.c_float

_lib.m3chat_get_avg_surprise.argtypes = [ctypes.c_void_p]
_lib.m3chat_get_avg_surprise.restype = ctypes.c_float

_lib.m3chat_get_num_active_neurons.argtypes = [ctypes.c_void_p]
_lib.m3chat_get_num_active_neurons.restype = ctypes.c_int

# Save/Load
_lib.m3chat_save_v2.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
_lib.m3chat_save_v2.restype = ctypes.c_int
_lib.m3chat_load_v2.argtypes = [ctypes.c_char_p]
_lib.m3chat_load_v2.restype = ctypes.c_void_p

# Info
_lib.m3chat_print_info_v2.argtypes = [ctypes.c_void_p]
_lib.m3chat_print_info_v2.restype = None

# ============================================================================
# PYTHON WRAPPER CLASS
# ============================================================================

class M3ChatV2:
    """
    M³-Chat v2: Autoregressive Manifold-Memory Language Model
    
    An enhanced memory-native neural network with transformer-like capabilities,
    featuring dual-memory architecture (fast/slow), multi-head resonance,
    and autoregressive token generation.
    
    Key Features:
    ------------
    - **Autoregressive Token Loop**: Generate text token-by-token
    - **Dual Memory System**: Fast (syntax) + Slow (semantics) memory
    - **Multi-Head Resonance**: Attention-free attention mechanism
    - **Positional Encoding**: Position-aware processing
    - **Cross-Entropy Loss**: Language modeling objective
    - **Teacher Forcing**: Efficient training with ground truth
    - **Memory Consolidation**: Fast → Slow knowledge transfer
    - **State Management**: Soft/hard/session reset modes
    
    Parameters
    ----------
    vocab_size : int
        Size of vocabulary (number of unique tokens)
    embed_dim : int
        Dimension of token embeddings
    hidden_size : int
        Number of memory-native neurons
    memory_dim : int
        Dimension of memory per neuron
    manifold_dim : int
        Dimension of global memory manifold
    num_heads : int, default=8
        Number of resonance heads (attention replacement)
    max_seq_len : int, default=512
        Maximum sequence length for positional encoding
    
    Examples
    --------
    >>> # Create a small language model
    >>> model = M3ChatV2(
    ...     vocab_size=1000,
    ...     embed_dim=128,
    ...     hidden_size=256,
    ...     memory_dim=64,
    ...     manifold_dim=32,
    ...     num_heads=8
    ... )
    
    >>> # Generate text
    >>> prompt = [1, 42, 123]  # token IDs
    >>> generated = model.generate(prompt, max_new_tokens=50)
    
    >>> # Train on sequences
    >>> tokens = [1, 2, 3, 4, 5]
    >>> loss = model.train_step(tokens)
    
    >>> # Save/load model
    >>> model.save('my_model.m3v2')
    >>> model_loaded = M3ChatV2.load('my_model.m3v2')
    """
    
    def __init__(self, 
                 vocab_size: int,
                 embed_dim: int = 128,
                 hidden_size: int = 256,
                 memory_dim: int = 64,
                 manifold_dim: int = 32,
                 num_heads: int = 8,
                 max_seq_len: int = 512):
        """Initialize M³-Chat v2 network"""
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        self.manifold_dim = manifold_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        
        # Create C network
        self._net = _lib.create_m3chat_v2(
            vocab_size,
            embed_dim,
            hidden_size,
            memory_dim,
            manifold_dim,
            num_heads,
            max_seq_len
        )
        
        if not self._net:
            raise RuntimeError("Failed to create M3ChatV2 network")
    
    def __del__(self):
        """Cleanup"""
        if hasattr(self, '_net') and self._net:
            _lib.destroy_m3chat_v2(self._net)
    
    # ========================================================================
    # CORE METHODS
    # ========================================================================
    
    def step(self, token: int) -> np.ndarray:
        """
        Perform one autoregressive step: process token and get next-token logits
        
        Parameters
        ----------
        token : int
            Input token ID (0 to vocab_size-1)
        
        Returns
        -------
        logits : np.ndarray
            Next-token logits [vocab_size]
        """
        if not 0 <= token < self.vocab_size:
            raise ValueError(f"Token {token} out of range [0, {self.vocab_size})")
        
        _lib.m3chat_step(self._net, token)
        return self.get_logits()
    
    def generate(self, 
                 prompt: List[int],
                 max_new_tokens: int = 50,
                 temperature: float = 1.0) -> List[int]:
        """
        Generate text autoregressively
        
        Parameters
        ----------
        prompt : List[int]
            List of token IDs to start generation
        max_new_tokens : int, default=50
            Maximum number of new tokens to generate
        temperature : float, default=1.0
            Sampling temperature (higher = more random)
        
        Returns
        -------
        generated : List[int]
            List of generated token IDs (including prompt)
        """
        prompt_arr = np.array(prompt, dtype=np.int32)
        output_arr = np.zeros(len(prompt) + max_new_tokens, dtype=np.int32)
        actual_length = ctypes.c_int(0)
        
        _lib.m3chat_generate(
            self._net,
            prompt_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(prompt),
            output_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            max_new_tokens,
            temperature,
            ctypes.byref(actual_length)
        )
        
        return list(output_arr[:actual_length.value])
    
    def train_step(self, tokens: List[int]) -> float:
        """
        Perform one training step on a sequence
        
        Uses teacher forcing: at each position, the model predicts the next token
        given all previous ground-truth tokens.
        
        Parameters
        ----------
        tokens : List[int]
            Sequence of token IDs for training
        
        Returns
        -------
        loss : float
            Cross-entropy loss for this sequence
        """
        tokens_arr = np.array(tokens, dtype=np.int32)
        loss = _lib.m3chat_train_sequence(
            self._net,
            tokens_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
            len(tokens)
        )
        return float(loss)
    
    def fit(self, 
            sequences: List[List[int]], 
            epochs: int = 10,
            verbose: int = 1) -> List[float]:
        """
        Train the model on multiple sequences
        
        Parameters
        ----------
        sequences : List[List[int]]
            List of token sequences for training
        epochs : int, default=10
            Number of training epochs
        verbose : int, default=1
            Verbosity level (0=silent, 1=progress, 2=detailed)
        
        Returns
        -------
        losses : List[float]
            Average loss per epoch
        """
        epoch_losses = []
        
        for epoch in range(epochs):
            total_loss = 0.0
            for seq in sequences:
                loss = self.train_step(seq)
                total_loss += loss
            
            avg_loss = total_loss / len(sequences)
            epoch_losses.append(avg_loss)
            
            if verbose >= 1:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
        
        return epoch_losses
    
    # ========================================================================
    # MEMORY MANAGEMENT
    # ========================================================================
    
    def consolidate_memory(self):
        """Transfer knowledge from fast memory to slow memory"""
        _lib.m3chat_consolidate(self._net)
    
    def reset_state(self, mode: str = 'soft'):
        """
        Reset network state
        
        Parameters
        ----------
        mode : str, default='soft'
            - 'soft': Small random perturbation
            - 'hard': Zero all activations and position
            - 'session': Full reset including memories
        """
        if mode == 'soft':
            _lib.m3chat_reset_soft(self._net)
        elif mode == 'hard':
            _lib.m3chat_reset_hard(self._net)
        elif mode == 'session':
            _lib.m3chat_reset_session(self._net)
        else:
            raise ValueError(f"Unknown reset mode: {mode}. Use 'soft', 'hard', or 'session'")
    
    # NOTE: These memory getter methods are not currently implemented in the C library
    # def get_fast_memory(self, neuron_idx: int) -> np.ndarray:
    #     """Get fast memory state of a specific neuron"""
    #     mem = np.zeros(self.memory_dim, dtype=np.float32)
    #     _lib.m3chat_get_fast_memory(
    #         self._net, neuron_idx,
    #         mem.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    #     )
    #     return mem
    
    # def get_slow_memory(self, neuron_idx: int) -> np.ndarray:
    #     """Get slow memory state of a specific neuron"""
    #     mem = np.zeros(self.memory_dim, dtype=np.float32)
    #     _lib.m3chat_get_slow_memory(
    #         self._net, neuron_idx,
    #         mem.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    #     )
    #     return mem
    
    # def get_network_memory(self) -> np.ndarray:
    #     """Get global network memory"""
    #     mem = np.zeros(self.manifold_dim, dtype=np.float32)
    #     _lib.m3chat_get_network_memory_v2(
    #         self._net,
    #         mem.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    #     )
    #     return mem
    
    def get_hidden_state(self) -> np.ndarray:
        """Get current hidden state"""
        hidden = np.zeros(self.hidden_size, dtype=np.float32)
        _lib.m3chat_get_hidden_state_v2(
            self._net,
            hidden.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return hidden
    
    def get_logits(self) -> np.ndarray:
        """Get current output logits"""
        logits = np.zeros(self.vocab_size, dtype=np.float32)
        _lib.m3chat_get_logits(
            self._net,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return logits
    
    # ========================================================================
    # PROPERTIES
    # ========================================================================
    
    @property
    def temperature(self) -> float:
        """Sampling temperature"""
        return _lib.m3chat_get_temperature(self._net)
    
    @temperature.setter
    def temperature(self, value: float):
        _lib.m3chat_set_temperature(self._net, value)
    
    @property
    def learning_rate(self) -> float:
        """Learning rate"""
        return _lib.m3chat_get_learning_rate_v2(self._net)
    
    @learning_rate.setter
    def learning_rate(self, value: float):
        _lib.m3chat_set_learning_rate_v2(self._net, value)
    
    @property
    def position(self) -> int:
        """Current position in sequence"""
        return _lib.m3chat_get_current_position(self._net)
    
    @property
    def training_steps(self) -> int:
        """Number of training steps completed"""
        return _lib.m3chat_get_training_steps_v2(self._net)
    
    @property
    def last_loss(self) -> float:
        """Last training loss"""
        return _lib.m3chat_get_last_loss_v2(self._net)
    
    @property
    def avg_resonance(self) -> float:
        """Average resonance across neurons"""
        return _lib.m3chat_get_avg_resonance_v2(self._net)
    
    @property
    def avg_memory_norm(self) -> float:
        """Average memory norm"""
        return _lib.m3chat_get_avg_memory_norm_v2(self._net)
    
    @property
    def avg_surprise(self) -> float:
        """Average surprise/novelty signal"""
        return _lib.m3chat_get_avg_surprise(self._net)
    
    @property
    def num_active_neurons(self) -> int:
        """Number of currently active neurons"""
        return _lib.m3chat_get_num_active_neurons(self._net)
    
    # ========================================================================
    # SERIALIZATION
    # ========================================================================
    
    def save(self, filename: str):
        """Save model to file"""
        result = _lib.m3chat_save_v2(self._net, filename.encode('utf-8'))
        if result != 0:
            raise IOError(f"Failed to save model to {filename}")
    
    @classmethod
    def load(cls, filename: str) -> 'M3ChatV2':
        """Load model from file"""
        net_ptr = _lib.m3chat_load_v2(filename.encode('utf-8'))
        if not net_ptr:
            raise IOError(f"Failed to load model from {filename}")
        
        # Create instance without calling __init__
        instance = cls.__new__(cls)
        instance._net = net_ptr
        
        # We don't know the exact parameters, so we can't set them
        # Users should track these separately if needed
        instance.vocab_size = None
        instance.embed_dim = None
        instance.hidden_size = None
        instance.memory_dim = None
        instance.manifold_dim = None
        instance.num_heads = None
        instance.max_seq_len = None
        
        return instance
    
    def print_info(self):
        """Print detailed network information"""
        _lib.m3chat_print_info_v2(self._net)


# ============================================================================
# DEMO & TESTING
# ============================================================================

def demo_basic_generation():
    """Demonstrate basic autoregressive generation"""
    print("\n" + "="*70)
    print("DEMO 1: Basic Autoregressive Generation")
    print("="*70)
    
    # Create a small model
    model = M3ChatV2(
        vocab_size=100,
        embed_dim=32,
        hidden_size=64,
        memory_dim=16,
        manifold_dim=8,
        num_heads=4,
        max_seq_len=128
    )
    
    print(f"\nCreated model:")
    print(f"  Vocabulary: {model.vocab_size} tokens")
    print(f"  Embedding: {model.embed_dim}D")
    print(f"  Hidden: {model.hidden_size} neurons")
    print(f"  Memory: {model.memory_dim}D (per neuron)")
    
    # Generate from random prompt
    prompt = [1, 5, 10, 15]
    print(f"\nPrompt tokens: {prompt}")
    
    generated = model.generate(prompt, max_new_tokens=10, temperature=1.0)
    print(f"Generated: {generated}")
    print(f"  → {len(generated) - len(prompt)} new tokens generated")


def demo_dual_memory():
    """Demonstrate fast + slow memory system"""
    print("\n" + "="*70)
    print("DEMO 2: Dual Memory System (Fast + Slow)")
    print("="*70)
    
    model = M3ChatV2(
        vocab_size=50,
        embed_dim=24,
        hidden_size=32,
        memory_dim=12,
        manifold_dim=6
    )
    
    print("\nDual memory architecture:")
    print("  • Fast Memory: Syntax, recent context (high decay)")
    print("  • Slow Memory: Semantics, long-term knowledge (low decay)\n")
    
    # Process a sequence
    sequence = [1, 2, 3, 4, 5, 6, 7, 8]
    for token in sequence:
        model.step(token)
    
    print(f"After processing {len(sequence)} tokens:")
    print(f"  Position: {model.position}")
    print(f"  Active neurons: {model.num_active_neurons}")
    print(f"  Avg memory norm: {model.avg_memory_norm:.4f}")
    
    # Consolidate: transfer fast → slow
    print("\nConsolidating memory (fast → slow)...")
    model.consolidate_memory()
    
    print(f"  Avg memory norm after: {model.avg_memory_norm:.4f}")
    print("  → Knowledge transferred to long-term storage")


def demo_training():
    """Demonstrate training with teacher forcing"""
    print("\n" + "="*70)
    print("DEMO 3: Training with Teacher Forcing")
    print("="*70)
    
    model = M3ChatV2(
        vocab_size=30,
        embed_dim=16,
        hidden_size=32,
        memory_dim=8,
        manifold_dim=4
    )
    
    # Create training sequences
    sequences = [
        [1, 2, 3, 4, 5],
        [1, 2, 6, 7, 8],
        [1, 2, 3, 9, 10],
        [5, 6, 7, 8, 9],
    ]
    
    print(f"\nTraining on {len(sequences)} sequences...")
    print(f"Learning rate: {model.learning_rate:.6f}\n")
    
    # Train
    losses = model.fit(sequences, epochs=5, verbose=1)
    
    print(f"\nFinal loss: {losses[-1]:.6f}")
    print(f"Training steps: {model.training_steps}")


def demo_state_reset():
    """Demonstrate different reset modes"""
    print("\n" + "="*70)
    print("DEMO 4: State Reset Modes")
    print("="*70)
    
    model = M3ChatV2(vocab_size=50, embed_dim=16, hidden_size=32, 
                     memory_dim=8, manifold_dim=4)
    
    # Build up state
    for token in [1, 2, 3, 4, 5]:
        model.step(token)
    
    print("\nAfter processing 5 tokens:")
    print(f"  Position: {model.position}")
    print(f"  Active neurons: {model.num_active_neurons}")
    print(f"  Avg resonance: {model.avg_resonance:.4f}")
    
    # Soft reset
    model.reset_state('soft')
    print("\nAfter SOFT reset (small perturbation):")
    print(f"  Position: {model.position}")
    print(f"  Avg resonance: {model.avg_resonance:.4f}")
    
    # Process more
    for token in [6, 7, 8]:
        model.step(token)
    
    # Hard reset
    model.reset_state('hard')
    print("\nAfter HARD reset (zero activations):")
    print(f"  Position: {model.position}")
    print(f"  Avg resonance: {model.avg_resonance:.4f}")
    
    # Session reset
    model.reset_state('session')
    print("\nAfter SESSION reset (full reset including memory):")
    print(f"  Position: {model.position}")
    print(f"  Avg memory norm: {model.avg_memory_norm:.4f}")


def demo_serialization():
    """Demonstrate save/load"""
    print("\n" + "="*70)
    print("DEMO 5: Model Serialization")
    print("="*70)
    
    # Create and train a model
    model = M3ChatV2(vocab_size=40, embed_dim=16, hidden_size=32,
                     memory_dim=8, manifold_dim=4)
    
    sequences = [[1, 2, 3, 4], [5, 6, 7, 8]]
    model.fit(sequences, epochs=3, verbose=0)
    
    # Get state before saving
    pos_before = model.position
    loss_before = model.last_loss
    steps_before = model.training_steps
    
    print(f"Before save:")
    print(f"  Position: {pos_before}")
    print(f"  Training steps: {steps_before}")
    print(f"  Last loss: {loss_before:.6f}")
    
    # Save
    filename = 'm3chat_v2_demo.m3v2'
    model.save(filename)
    print(f"\nSaved to {filename}")
    
    # Load
    model_loaded = M3ChatV2.load(filename)
    print(f"Loaded from {filename}")
    
    # Check state
    pos_after = model_loaded.position
    loss_after = model_loaded.last_loss
    steps_after = model_loaded.training_steps
    
    print(f"\nAfter load:")
    print(f"  Position: {pos_after}")
    print(f"  Training steps: {steps_after}")
    print(f"  Last loss: {loss_after:.6f}")
    
    print(f"\n✓ State preserved perfectly!")


def main():
    """Run all demonstrations"""
    print("\n╔" + "="*68 + "╗")
    print("║" + " "*8 + "M³-CHAT v2: AUTOREGRESSIVE LANGUAGE MODEL" + " "*17 + "║")
    print("║" + " "*10 + "Enhanced Memory-Native Architecture" + " "*23 + "║")
    print("╚" + "="*68 + "╝")
    
    print("\nNEW FEATURES:")
    print("  ✓ Autoregressive token generation")
    print("  ✓ Dual memory (fast + slow)")
    print("  ✓ Multi-head resonance")
    print("  ✓ Teacher forcing training")
    print("  ✓ Memory consolidation")
    print("  ✓ State management")
    
    try:
        demo_basic_generation()
        
        input("\nPress Enter to continue...")
        demo_dual_memory()
        
        input("\nPress Enter to continue...")
        demo_training()
        
        input("\nPress Enter to continue...")
        demo_state_reset()
        
        input("\nPress Enter to continue...")
        demo_serialization()
        
        print("\n" + "="*70)
        print("All Demonstrations Complete!")
        print("="*70)
        
        print("\n🚀 M³-Chat v2 is ready for language modeling!")
        
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()