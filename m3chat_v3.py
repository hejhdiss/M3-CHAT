"""
M³-CHAT v3 Python Wrapper
GPT-Competitive Memory-Native Neural Network with ALL Critical Upgrades

CRITICAL UPGRADES:
✅ 1. Query-Key-Value memory (like transformers)
✅ 2. Causal time gating (prevents future leakage)
✅ 3. Deep FFN reasoning blocks (4x expansion)
✅ 4. RMSNorm + gated residuals (stable scaling)
✅ 5. Multi-step internal recurrence (thinking loop)
✅ 6. Read-write memory separation (prevents forgetting)
✅ 7. Global scratchpad state (planning buffer)

Usage:
    from m3chat_v3 import M3ChatV3
    
    # Create model with thinking capability
    model = M3ChatV3(
        vocab_size=10000,
        embed_dim=128,
        hidden_size=256,
        memory_dim=64,
        manifold_dim=32,
        num_heads=4,
        max_seq_len=512,
        n_thoughts=3  # NEW: internal thinking steps
    )
    
    # Train
    tokens = [1, 2, 3, 4, 5]
    loss = model.train_sequence(tokens)
    
    # Generate
    prompt = [1, 2, 3]
    output = model.generate(prompt, max_new_tokens=20, temperature=0.8)
    
    # Inspect internal state
    scratchpad = model.get_scratchpad()  # NEW: planning buffer

LICENSED UDNER GPL V3.
"""

import ctypes
import numpy as np
import os
import platform

class M3ChatV3:
    def __init__(
        self,
        vocab_size=10000,
        embed_dim=128,
        hidden_size=256,
        memory_dim=64,
        manifold_dim=32,
        num_heads=4,
        max_seq_len=512,
        n_thoughts=2,  # NEW: number of internal thinking steps
        lib_path=None
    ):
        """
        Initialize M³-Chat v3 with all architectural upgrades.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Token embedding dimension
            hidden_size: Number of memory-native neurons
            memory_dim: Memory dimension per neuron
            manifold_dim: Global manifold dimension (and scratchpad size)
            num_heads: Number of attention heads for Q-K-V mechanism
            max_seq_len: Maximum sequence length
            n_thoughts: Number of internal thinking iterations (NEW)
            lib_path: Path to compiled library (auto-detected if None)
        """
        # Load library
        if lib_path is None:
            lib_path = self._find_library()
        
        self.lib = ctypes.CDLL(lib_path)
        
        # Define function signatures
        self._setup_function_signatures()
        
        # Store architecture parameters
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_size = hidden_size
        self.memory_dim = memory_dim
        self.manifold_dim = manifold_dim
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.n_thoughts = n_thoughts
        
        # Create network
        self.net = self.lib.create_m3chat_v3(
            vocab_size,
            embed_dim,
            hidden_size,
            memory_dim,
            manifold_dim,
            num_heads,
            max_seq_len,
            n_thoughts
        )
        
        if not self.net:
            raise RuntimeError("Failed to create M3ChatV3 network")
    
    def _find_library(self):
        """Auto-detect compiled library based on platform."""
        system = platform.system()
        print(system)
        
        candidates = []
        base_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'm3chat_v3.dll')
        if system == "Windows":
            candidates = ["m3chat_v3.dll", "./m3chat_v3.dll", "../m3chat_v3.dll",base_path]
        elif system == "Darwin":  # macOS
            candidates = ["m3chat_v3.dylib", "./m3chat_v3.dylib", "../m3chat_v3.dylib"]
        else:  # Linux
            candidates = ["m3chat_v3.so", "./m3chat_v3.so", "../m3chat_v3.so"]
        
        for path in candidates:
            if os.path.exists(path):
                return path
        
        raise FileNotFoundError(
            f"Could not find M3ChatV3 library. Please compile m3chat_v3.c first.\n"
        )
    
    def _setup_function_signatures(self):
        """Define C function signatures."""
        # Creation and destruction
        self.lib.create_m3chat_v3.argtypes = [
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
        ]
        self.lib.create_m3chat_v3.restype = ctypes.c_void_p
        
        self.lib.destroy_m3chat_v3.argtypes = [ctypes.c_void_p]
        self.lib.destroy_m3chat_v3.restype = None
        
        # Forward pass
        self.lib.m3chat_step_v3.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.m3chat_step_v3.restype = None
        
        # Sampling
        self.lib.m3chat_sample_v3.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.lib.m3chat_sample_v3.restype = ctypes.c_int
        
        self.lib.m3chat_argmax_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_argmax_v3.restype = ctypes.c_int
        
        # Training
        self.lib.m3chat_train_sequence_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_bool
        ]
        self.lib.m3chat_train_sequence_v3.restype = ctypes.c_float
        
        # Generation
        self.lib.m3chat_generate_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.m3chat_generate_v3.restype = None
        
        # State reset
        self.lib.m3chat_reset_soft_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_reset_soft_v3.restype = None
        
        self.lib.m3chat_reset_hard_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_reset_hard_v3.restype = None
        
        self.lib.m3chat_reset_session_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_reset_session_v3.restype = None
        
        # Getters
        self.lib.m3chat_get_logits_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.m3chat_get_logits_v3.restype = None
        
        self.lib.m3chat_get_hidden_state_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.m3chat_get_hidden_state_v3.restype = None
        
        # NEW: Scratchpad getter
        self.lib.m3chat_get_scratchpad_v3.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float)
        ]
        self.lib.m3chat_get_scratchpad_v3.restype = None
        
        # Setters
        self.lib.m3chat_set_temperature_v3.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.lib.m3chat_set_temperature_v3.restype = None
        
        self.lib.m3chat_set_learning_rate_v3.argtypes = [ctypes.c_void_p, ctypes.c_float]
        self.lib.m3chat_set_learning_rate_v3.restype = None
        
        # NEW: Thinking steps control
        self.lib.m3chat_set_n_thoughts_v3.argtypes = [ctypes.c_void_p, ctypes.c_int]
        self.lib.m3chat_set_n_thoughts_v3.restype = None
        
        self.lib.m3chat_get_n_thoughts_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_get_n_thoughts_v3.restype = ctypes.c_int
        
        # Statistics getters
        self.lib.m3chat_get_temperature_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_get_temperature_v3.restype = ctypes.c_float
        
        self.lib.m3chat_get_learning_rate_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_get_learning_rate_v3.restype = ctypes.c_float
        
        self.lib.m3chat_get_training_steps_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_get_training_steps_v3.restype = ctypes.c_int
        
        self.lib.m3chat_get_last_loss_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_get_last_loss_v3.restype = ctypes.c_float
        
        self.lib.m3chat_get_avg_resonance_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_get_avg_resonance_v3.restype = ctypes.c_float
        
        self.lib.m3chat_get_avg_memory_norm_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_get_avg_memory_norm_v3.restype = ctypes.c_float
        
        self.lib.m3chat_get_avg_surprise_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_get_avg_surprise_v3.restype = ctypes.c_float
        
        self.lib.m3chat_get_num_active_neurons_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_get_num_active_neurons_v3.restype = ctypes.c_int
        
        self.lib.m3chat_get_current_position_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_get_current_position_v3.restype = ctypes.c_int
        
        # Serialization
        self.lib.m3chat_save_v3.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.m3chat_save_v3.restype = ctypes.c_int
        
        self.lib.m3chat_load_v3.argtypes = [ctypes.c_char_p]
        self.lib.m3chat_load_v3.restype = ctypes.c_void_p
        
        # Info
        self.lib.m3chat_print_info_v3.argtypes = [ctypes.c_void_p]
        self.lib.m3chat_print_info_v3.restype = None
    
    def step(self, token_id):
        """
        Single autoregressive step with ALL v3 upgrades:
        - Q-K-V attention mechanism
        - Time-decayed causal gating
        - Deep FFN reasoning
        - Multi-step internal thinking
        - Read-write memory separation
        """
        self.lib.m3chat_step_v3(self.net, int(token_id))
    
    def sample(self, temperature=1.0):
        """Sample next token from current logits."""
        return self.lib.m3chat_sample_v3(self.net, float(temperature))
    
    def argmax(self):
        """Get most likely next token."""
        return self.lib.m3chat_argmax_v3(self.net)
    
    def train_sequence(self, tokens, use_teacher_forcing=True):
        """
        Train on a sequence of tokens.
        
        Args:
            tokens: List or array of token IDs
            use_teacher_forcing: Whether to use teacher forcing
            
        Returns:
            Average loss over the sequence
        """
        tokens_array = (ctypes.c_int * len(tokens))(*tokens)
        return self.lib.m3chat_train_sequence_v3(
            self.net,
            tokens_array,
            len(tokens),
            use_teacher_forcing
        )
    
    def generate(self, prompt_tokens, max_new_tokens=50, temperature=1.0):
        """
        Generate text continuation.
        
        Args:
            prompt_tokens: List of prompt token IDs
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            List of generated token IDs (including prompt)
        """
        prompt_array = (ctypes.c_int * len(prompt_tokens))(*prompt_tokens)
        output_array = (ctypes.c_int * (len(prompt_tokens) + max_new_tokens))()
        actual_length = ctypes.c_int()
        
        self.lib.m3chat_generate_v3(
            self.net,
            prompt_array,
            len(prompt_tokens),
            output_array,
            max_new_tokens,
            float(temperature),
            ctypes.byref(actual_length)
        )
        
        return list(output_array[:actual_length.value])
    
    def reset_soft(self):
        """Soft reset: decay fast memory, keep slow memory."""
        self.lib.m3chat_reset_soft_v3(self.net)
    
    def reset_hard(self):
        """Hard reset: clear all memory."""
        self.lib.m3chat_reset_hard_v3(self.net)
    
    def reset_session(self):
        """Session reset: clear read memory, keep write memory (knowledge)."""
        self.lib.m3chat_reset_session_v3(self.net)
    
    def get_logits(self):
        """Get current output logits."""
        logits = np.zeros(self.vocab_size, dtype=np.float32)
        self.lib.m3chat_get_logits_v3(
            self.net,
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return logits
    
    def get_hidden_state(self):
        """Get current hidden state."""
        state = np.zeros(self.hidden_size, dtype=np.float32)
        self.lib.m3chat_get_hidden_state_v3(
            self.net,
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return state
    
    def get_scratchpad(self):
        """
        NEW: Get internal scratchpad/planning buffer.
        This shows the network's internal "thoughts" before producing output.
        """
        scratchpad = np.zeros(self.manifold_dim, dtype=np.float32)
        self.lib.m3chat_get_scratchpad_v3(
            self.net,
            scratchpad.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        return scratchpad
    
    def get_probabilities(self, temperature=1.0):
        """Get token probabilities (softmax of logits)."""
        logits = self.get_logits()
        # Numerical stability
        logits = logits - np.max(logits)
        exp_logits = np.exp(logits / temperature)
        return exp_logits / np.sum(exp_logits)
    
    # Parameter setters/getters
    
    def set_temperature(self, temp):
        """Set sampling temperature."""
        self.lib.m3chat_set_temperature_v3(self.net, float(temp))
    
    def get_temperature(self):
        """Get current temperature."""
        return self.lib.m3chat_get_temperature_v3(self.net)
    
    def set_learning_rate(self, lr):
        """Set learning rate."""
        self.lib.m3chat_set_learning_rate_v3(self.net, float(lr))
    
    def get_learning_rate(self):
        """Get current learning rate."""
        return self.lib.m3chat_get_learning_rate_v3(self.net)
    
    def set_n_thoughts(self, n):
        """
        NEW: Set number of internal thinking iterations.
        Higher values = more deliberate reasoning, but slower.
        """
        self.lib.m3chat_set_n_thoughts_v3(self.net, int(n))
    
    def get_n_thoughts(self):
        """NEW: Get current number of thinking iterations."""
        return self.lib.m3chat_get_n_thoughts_v3(self.net)
    
    # Statistics
    
    def get_training_steps(self):
        """Get total training steps."""
        return self.lib.m3chat_get_training_steps_v3(self.net)
    
    def get_last_loss(self):
        """Get loss from last training step."""
        return self.lib.m3chat_get_last_loss_v3(self.net)
    
    def get_avg_resonance(self):
        """Get average resonance (attention-like scores)."""
        return self.lib.m3chat_get_avg_resonance_v3(self.net)
    
    def get_avg_memory_norm(self):
        """Get average memory norm."""
        return self.lib.m3chat_get_avg_memory_norm_v3(self.net)
    
    def get_avg_surprise(self):
        """Get average surprise signal."""
        return self.lib.m3chat_get_avg_surprise_v3(self.net)
    
    def get_num_active_neurons(self):
        """Get number of currently active neurons."""
        return self.lib.m3chat_get_num_active_neurons_v3(self.net)
    
    def get_current_position(self):
        """Get current position in sequence."""
        return self.lib.m3chat_get_current_position_v3(self.net)
    
    def get_stats(self):
        """Get all statistics as a dictionary."""
        return {
            'position': self.get_current_position(),
            'training_steps': self.get_training_steps(),
            'last_loss': self.get_last_loss(),
            'avg_resonance': self.get_avg_resonance(),
            'avg_memory_norm': self.get_avg_memory_norm(),
            'avg_surprise': self.get_avg_surprise(),
            'num_active_neurons': self.get_num_active_neurons(),
            'temperature': self.get_temperature(),
            'learning_rate': self.get_learning_rate(),
            'n_thoughts': self.get_n_thoughts()
        }
    
    # Serialization
    
    def save(self, filename):
        """Save model to file."""
        result = self.lib.m3chat_save_v3(self.net, filename.encode('utf-8'))
        if result != 0:
            raise RuntimeError(f"Failed to save model to {filename}")
    
    @classmethod
    def load(cls, filename, lib_path=None):
        """
        Load model from file.
        
        Args:
            filename: Path to saved model
            lib_path: Path to library (auto-detected if None)
            
        Returns:
            M3ChatV3 instance
        """
        # Create empty instance
        instance = cls.__new__(cls)
        
        # Load library
        if lib_path is None:
            lib_path = instance._find_library()
        
        instance.lib = ctypes.CDLL(lib_path)
        instance._setup_function_signatures()
        
        # Load network
        instance.net = instance.lib.m3chat_load_v3(filename.encode('utf-8'))
        if not instance.net:
            raise RuntimeError(f"Failed to load model from {filename}")
        
        # We don't know the exact architecture from file, so set placeholder values
        # In practice, you'd want to store these in the file header
        instance.vocab_size = None
        instance.embed_dim = None
        instance.hidden_size = None
        instance.memory_dim = None
        instance.manifold_dim = None
        instance.num_heads = None
        instance.max_seq_len = None
        instance.n_thoughts = None
        
        return instance
    
    def print_info(self):
        """Print detailed model information."""
        self.lib.m3chat_print_info_v3(self.net)
    
    def __del__(self):
        """Cleanup on deletion."""
        if hasattr(self, 'net') and self.net:
            self.lib.destroy_m3chat_v3(self.net)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("M³-CHAT v3 Demo: GPT-Competitive Architecture")
    print("=" * 70)
    
    # Create model
    print("\n1. Creating model with all upgrades...")
    model = M3ChatV3(
        vocab_size=100,
        embed_dim=64,
        hidden_size=128,
        memory_dim=32,
        manifold_dim=16,
        num_heads=4,
        max_seq_len=256,
        n_thoughts=3  # 3 internal thinking iterations
    )
    
    model.print_info()
    
    # Training example
    print("\n2. Training on simple sequence...")
    training_sequences = [
        [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        [1, 3, 5, 7, 9, 2, 4, 6, 8, 10],
    ]
    
    model.set_learning_rate(0.01)
    
    for epoch in range(5):
        total_loss = 0.0
        for seq in training_sequences:
            loss = model.train_sequence(seq)
            total_loss += loss
        
        avg_loss = total_loss / len(training_sequences)
        print(f"  Epoch {epoch+1}: Loss = {avg_loss:.4f}")
    
    # Generation example
    print("\n3. Generating with internal thinking...")
    prompt = [1, 2, 3]
    
    # Try different thinking depths
    for n_thoughts in [1, 2, 3]:
        model.set_n_thoughts(n_thoughts)
        model.reset_session()
        
        output = model.generate(prompt, max_new_tokens=7, temperature=0.8)
        print(f"  {n_thoughts} thoughts: {output}")
    
    # Inspect internal state
    print("\n4. Inspecting internal state...")
    model.reset_session()
    model.step(1)
    model.step(2)
    model.step(3)
    
    stats = model.get_stats()
    print(f"  Position: {stats['position']}")
    print(f"  Active neurons: {stats['num_active_neurons']}")
    print(f"  Avg resonance: {stats['avg_resonance']:.4f}")
    print(f"  Avg surprise: {stats['avg_surprise']:.4f}")
    
    scratchpad = model.get_scratchpad()
    print(f"  Scratchpad norm: {np.linalg.norm(scratchpad):.4f}")
    print(f"  Scratchpad top-3 dims: {np.argsort(np.abs(scratchpad))[-3:]}")
    
    # Save model
    print("\n5. Saving model...")
    model.save("m3chat_v3_demo.bin")
    print("  Saved to m3chat_v3_demo.bin")
    
    # Load model
    print("\n6. Loading model...")
    loaded_model = M3ChatV3.load("m3chat_v3_demo.bin")
    loaded_model.print_info()
    
    print("\n" + "=" * 70)
    print("Demo complete!")
    print("=" * 70)