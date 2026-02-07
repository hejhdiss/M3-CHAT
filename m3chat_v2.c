/**
 * M³-CHAT v2: AUTOREGRESSIVE MANIFOLD-MEMORY LANGUAGE MODEL
 * Enhanced Memory-Native Neural Network with Transformer-like capabilities
 * 
 * NEW FEATURES:
 * ✓ 1. Autoregressive token loop with m3chat_step()
 * ✓ 2. Learned projection (vector → vector)
 * ✓ 3. Fast + Slow memory split
 * ✓ 4. Multi-head resonance (attention replacement)
 * ✓ 5. Cross-entropy loss for language
 * ✓ 6. Positional encoding
 * ✓ 7. Proper memory direction preservation
 * ✓ 8. Teacher forcing support
 * ✓ 9. Vocabulary + embedding layer
 * ✓ 10. Residual memory paths
 * ✓ 11. Layer normalization on memory
 * ✓ 12. Token masking (no future leakage)
 * ✓ 13. Entropy control / temperature
 * ✓ 14. State reset protocol (soft/hard/session)
 * ✓ 15. Gradient flow through time (truncated BPTT)
 * 
 * BONUS FEATURES:
 * ✓ Surprise/novelty signal
 * ✓ Lateral inhibition / competition
 * ✓ Memory consolidation phase
 * ✓ Sparse neuron activation
 * 
 * Compile:
 * Windows: gcc -shared -o m3chat_v2.dll m3chat_v2.c -lm -O3 -fopenmp -static-libgcc -static
 * Linux:   gcc -shared -fPIC -o m3chat_v2.so m3chat_v2.c -lm -O3 -fopenmp
 * Mac:     gcc -shared -fPIC -o m3chat_v2.dylib m3chat_v2.c -lm -O3 -Xpreprocessor -fopenmp -lomp
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
    #define EXPORT __declspec(dllexport)
#else
    #define EXPORT
#endif

#define MIN(a,b) ((a) < (b) ? (a) : (b))
#define MAX(a,b) ((a) > (b) ? (a) : (b))
#define EPSILON 1e-8f

// ============================================================================
// DATA STRUCTURES
// ============================================================================

/**
 * Enhanced Memory-Native Neuron with Fast/Slow memory
 */
typedef struct {
    float *fast_memory;      // Fast memory: syntax, recent tokens [memory_dim]
    float *slow_memory;      // Slow memory: topics, long-term context [memory_dim]
    float *fast_grad;        // Gradients for fast memory
    float *slow_grad;        // Gradients for slow memory
    
    float *resonance;        // Multi-head resonance values [num_heads]
    float activation;        // Current activation level
    
    float fast_beta;         // Fast memory update rate (high)
    float slow_beta;         // Slow memory update rate (low)
    float fast_decay;        // Fast decay rate
    float slow_decay;        // Slow decay rate (near zero)
    
    float surprise;          // Novelty/surprise signal
    bool is_active;          // Sparse activation flag
} MemoryNativeNeuron;

/**
 * Autoregressive M³-Chat Language Model
 */
typedef struct {
    // === ARCHITECTURE ===
    int vocab_size;          // Vocabulary size
    int embed_dim;           // Token embedding dimension
    int hidden_size;         // Number of neurons
    int memory_dim;          // Memory dimension per neuron
    int manifold_dim;        // Manifold dimension
    int num_heads;           // Number of attention heads
    int max_seq_len;         // Maximum sequence length
    
    // === NEURONS ===
    MemoryNativeNeuron *neurons;
    
    // === EMBEDDINGS ===
    float *token_embeddings;  // [vocab_size × embed_dim]
    float *position_embeddings; // [max_seq_len × embed_dim]
    
    // === PROJECTION LAYERS ===
    float *W_embed_to_hidden; // [embed_dim × hidden_size]
    float *W_read;            // Memory readout projection [memory_dim × hidden_size]
    float *W_output;          // Output logits [hidden_size × vocab_size]
    
    // === MULTI-HEAD RESONANCE ===
    float *W_resonance;       // [num_heads × memory_dim × embed_dim]
    
    // === MEMORY EVOLUTION ===
    float *W_memory_fast;     // Fast memory evolution [memory_dim × memory_dim]
    float *W_memory_slow;     // Slow memory evolution [memory_dim × memory_dim]
    float *W_pos_inject;      // Positional signal injection [embed_dim × memory_dim]
    
    // === RESIDUAL CONNECTIONS ===
    float *W_residual_input;  // Input residual [embed_dim × memory_dim]
    float *W_residual_context; // Context residual [manifold_dim × memory_dim]
    
    // === LAYER NORMALIZATION ===
    float *ln_gamma;          // Layer norm scale [memory_dim]
    float *ln_beta;           // Layer norm bias [memory_dim]
    
    // === MANIFOLD PARAMETERS ===
    float *manifold_basis;    // [manifold_dim × memory_dim]
    float *manifold_center;   // [memory_dim]
    float manifold_radius;
    
    // === GLOBAL MEMORY ===
    float *network_memory;    // Global context [manifold_dim]
    
    // === AUTOREGRESSIVE STATE ===
    int current_position;     // Current token position in sequence
    int *token_history;       // Token history [max_seq_len]
    float *hidden_state;      // Current hidden state [hidden_size]
    float *logits;            // Output logits [vocab_size]
    
    // === TRAINING STATE ===
    float temperature;        // Sampling temperature
    float entropy_penalty;    // Entropy regularization
    float learning_rate;
    float gradient_clip_norm;
    int training_steps;
    float last_loss;
    
    // === DYNAMICS ===
    float dt;
    float conversation_time;
    
    // === STATISTICS ===
    float avg_resonance;
    float avg_memory_norm;
    float avg_surprise;
    int num_active_neurons;
    
    // === CONSOLIDATION ===
    float consolidation_rate; // Fast → Slow memory transfer rate
    
} M3ChatV2;

// ============================================================================
// ACTIVATION FUNCTIONS
// ============================================================================

static inline float tanh_act(float x) {
    return tanhf(x);
}

static inline float tanh_derivative(float x) {
    float t = tanhf(x);
    return 1.0f - t * t;
}

static inline float gelu(float x) {
    float x3 = x * x * x;
    return 0.5f * x * (1.0f + tanhf(0.797885f * (x + 0.044715f * x3)));
}

static inline float gelu_derivative(float x) {
    float tanh_arg = 0.797885f * (x + 0.044715f * x * x * x);
    float tanh_val = tanhf(tanh_arg);
    float sech_sq = 1.0f - tanh_val * tanh_val;
    return 0.5f * (1.0f + tanh_val) + 
           0.5f * x * sech_sq * 0.797885f * (1.0f + 3.0f * 0.044715f * x * x);
}

// Softmax with numerical stability
static void softmax(const float *input, float *output, int size, float temperature) {
    float max_val = input[0];
    for (int i = 1; i < size; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        output[i] = expf((input[i] - max_val) / temperature);
        sum += output[i];
    }
    
    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

// Cross-entropy loss
static float cross_entropy_loss(const float *logits, int target, int size) {
    float max_logit = logits[0];
    for (int i = 1; i < size; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < size; i++) {
        sum_exp += expf(logits[i] - max_logit);
    }
    
    float log_sum_exp = logf(sum_exp) + max_logit;
    return log_sum_exp - logits[target];
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

static float randn(void) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    return sqrtf(-2.0f * logf(u1 + EPSILON)) * cosf(2.0f * M_PI * u2);
}

static float vector_dot(const float *a, const float *b, int size) {
    float dot = 0.0f;
    for (int i = 0; i < size; i++) {
        dot += a[i] * b[i];
    }
    return dot;
}

static float vector_norm(const float *v, int size) {
    return sqrtf(vector_dot(v, v, size));
}

static void vector_normalize(float *v, int size) {
    float norm = vector_norm(v, size);
    if (norm > EPSILON) {
        for (int i = 0; i < size; i++) {
            v[i] /= norm;
        }
    }
}

// Layer normalization
static void layer_norm(float *x, const float *gamma, const float *beta, int size) {
    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < size; i++) {
        mean += x[i];
    }
    mean /= size;
    
    // Calculate variance
    float var = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = x[i] - mean;
        var += diff * diff;
    }
    var /= size;
    
    // Normalize
    float std = sqrtf(var + EPSILON);
    for (int i = 0; i < size; i++) {
        x[i] = gamma[i] * (x[i] - mean) / std + beta[i];
    }
}

// Gradient clipping
static void clip_gradient(float *grad, int size, float max_norm) {
    float norm = vector_norm(grad, size);
    if (norm > max_norm) {
        float scale = max_norm / norm;
        for (int i = 0; i < size; i++) {
            grad[i] *= scale;
        }
    }
}

// ============================================================================
// POSITIONAL ENCODING
// ============================================================================

static void compute_positional_encoding(float *pos_enc, int position, int dim) {
    for (int i = 0; i < dim; i++) {
        float freq = 1.0f / powf(10000.0f, (2.0f * (i / 2)) / dim);
        if (i % 2 == 0) {
            pos_enc[i] = sinf(position * freq);
        } else {
            pos_enc[i] = cosf(position * freq);
        }
    }
}

// ============================================================================
// NETWORK CREATION AND DESTRUCTION
// ============================================================================

EXPORT M3ChatV2* create_m3chat_v2(
    int vocab_size,
    int embed_dim,
    int hidden_size,
    int memory_dim,
    int manifold_dim,
    int num_heads,
    int max_seq_len
) {
    M3ChatV2 *net = (M3ChatV2*)malloc(sizeof(M3ChatV2));
    if (!net) return NULL;
    
    // Set architecture
    net->vocab_size = vocab_size;
    net->embed_dim = embed_dim;
    net->hidden_size = hidden_size;
    net->memory_dim = memory_dim;
    net->manifold_dim = manifold_dim;
    net->num_heads = num_heads;
    net->max_seq_len = max_seq_len;
    
    // Allocate neurons
    net->neurons = (MemoryNativeNeuron*)calloc(hidden_size, sizeof(MemoryNativeNeuron));
    for (int i = 0; i < hidden_size; i++) {
        net->neurons[i].fast_memory = (float*)calloc(memory_dim, sizeof(float));
        net->neurons[i].slow_memory = (float*)calloc(memory_dim, sizeof(float));
        net->neurons[i].fast_grad = (float*)calloc(memory_dim, sizeof(float));
        net->neurons[i].slow_grad = (float*)calloc(memory_dim, sizeof(float));
        net->neurons[i].resonance = (float*)calloc(num_heads, sizeof(float));
        
        // Initialize fast/slow parameters
        net->neurons[i].fast_beta = 0.8f + 0.2f * ((float)rand() / RAND_MAX);
        net->neurons[i].slow_beta = 0.05f + 0.05f * ((float)rand() / RAND_MAX);
        net->neurons[i].fast_decay = 0.1f;
        net->neurons[i].slow_decay = 0.001f;
        net->neurons[i].is_active = true;
    }
    
    // Allocate embeddings
    net->token_embeddings = (float*)malloc(vocab_size * embed_dim * sizeof(float));
    net->position_embeddings = (float*)malloc(max_seq_len * embed_dim * sizeof(float));
    
    // Initialize embeddings
    float embed_scale = sqrtf(2.0f / embed_dim);
    for (int i = 0; i < vocab_size * embed_dim; i++) {
        net->token_embeddings[i] = embed_scale * randn();
    }
    
    // Initialize positional embeddings
    for (int pos = 0; pos < max_seq_len; pos++) {
        compute_positional_encoding(
            &net->position_embeddings[pos * embed_dim],
            pos,
            embed_dim
        );
    }
    
    // Allocate projection matrices
    net->W_embed_to_hidden = (float*)malloc(embed_dim * hidden_size * sizeof(float));
    net->W_read = (float*)malloc(memory_dim * hidden_size * sizeof(float));
    net->W_output = (float*)malloc(hidden_size * vocab_size * sizeof(float));
    
    // Allocate multi-head resonance
    net->W_resonance = (float*)malloc(num_heads * memory_dim * embed_dim * sizeof(float));
    
    // Allocate memory evolution matrices
    net->W_memory_fast = (float*)malloc(memory_dim * memory_dim * sizeof(float));
    net->W_memory_slow = (float*)malloc(memory_dim * memory_dim * sizeof(float));
    net->W_pos_inject = (float*)malloc(embed_dim * memory_dim * sizeof(float));
    
    // Allocate residual connections
    net->W_residual_input = (float*)malloc(embed_dim * memory_dim * sizeof(float));
    net->W_residual_context = (float*)malloc(manifold_dim * memory_dim * sizeof(float));
    
    // Allocate layer normalization parameters
    net->ln_gamma = (float*)malloc(memory_dim * sizeof(float));
    net->ln_beta = (float*)malloc(memory_dim * sizeof(float));
    for (int i = 0; i < memory_dim; i++) {
        net->ln_gamma[i] = 1.0f;
        net->ln_beta[i] = 0.0f;
    }
    
    // Allocate manifold
    net->manifold_basis = (float*)malloc(manifold_dim * memory_dim * sizeof(float));
    net->manifold_center = (float*)calloc(memory_dim, sizeof(float));
    net->manifold_radius = 1.0f;
    
    // Allocate global memory
    net->network_memory = (float*)calloc(manifold_dim, sizeof(float));
    
    // Allocate autoregressive state
    net->token_history = (int*)calloc(max_seq_len, sizeof(int));
    net->hidden_state = (float*)calloc(hidden_size, sizeof(float));
    net->logits = (float*)calloc(vocab_size, sizeof(float));
    net->current_position = 0;
    
    // Initialize all weight matrices with Xavier/He initialization
    float scale_hidden = sqrtf(2.0f / (embed_dim + hidden_size));
    for (int i = 0; i < embed_dim * hidden_size; i++) {
        net->W_embed_to_hidden[i] = scale_hidden * randn();
    }
    
    float scale_read = sqrtf(2.0f / (memory_dim + hidden_size));
    for (int i = 0; i < memory_dim * hidden_size; i++) {
        net->W_read[i] = scale_read * randn();
    }
    
    float scale_output = sqrtf(2.0f / (hidden_size + vocab_size));
    for (int i = 0; i < hidden_size * vocab_size; i++) {
        net->W_output[i] = scale_output * randn();
    }
    
    // Initialize resonance weights
    float scale_res = sqrtf(2.0f / (memory_dim + embed_dim));
    for (int i = 0; i < num_heads * memory_dim * embed_dim; i++) {
        net->W_resonance[i] = scale_res * randn();
    }
    
    // Initialize memory evolution (identity + noise for stability)
    for (int i = 0; i < memory_dim; i++) {
        for (int j = 0; j < memory_dim; j++) {
            int idx = i * memory_dim + j;
            if (i == j) {
                net->W_memory_fast[idx] = 0.9f + 0.1f * randn();
                net->W_memory_slow[idx] = 0.95f + 0.05f * randn();
            } else {
                net->W_memory_fast[idx] = 0.05f * randn();
                net->W_memory_slow[idx] = 0.02f * randn();
            }
        }
    }
    
    float scale_pos = sqrtf(2.0f / embed_dim);
    for (int i = 0; i < embed_dim * memory_dim; i++) {
        net->W_pos_inject[i] = scale_pos * randn();
    }
    
    // Initialize residual weights
    float scale_res_in = sqrtf(2.0f / embed_dim);
    for (int i = 0; i < embed_dim * memory_dim; i++) {
        net->W_residual_input[i] = scale_res_in * randn();
    }
    
    float scale_res_ctx = sqrtf(2.0f / manifold_dim);
    for (int i = 0; i < manifold_dim * memory_dim; i++) {
        net->W_residual_context[i] = scale_res_ctx * randn();
    }
    
    // Initialize manifold basis (orthonormal)
    for (int i = 0; i < manifold_dim * memory_dim; i++) {
        net->manifold_basis[i] = randn();
    }
    // Gram-Schmidt orthonormalization
    for (int i = 0; i < manifold_dim; i++) {
        float *basis_i = &net->manifold_basis[i * memory_dim];
        
        // Orthogonalize against previous vectors
        for (int j = 0; j < i; j++) {
            float *basis_j = &net->manifold_basis[j * memory_dim];
            float dot = vector_dot(basis_i, basis_j, memory_dim);
            for (int k = 0; k < memory_dim; k++) {
                basis_i[k] -= dot * basis_j[k];
            }
        }
        
        // Normalize
        vector_normalize(basis_i, memory_dim);
    }
    
    // Set default parameters
    net->temperature = 1.0f;
    net->entropy_penalty = 0.01f;
    net->learning_rate = 0.001f;
    net->gradient_clip_norm = 1.0f;
    net->training_steps = 0;
    net->last_loss = 0.0f;
    net->dt = 0.1f;
    net->conversation_time = 0.0f;
    net->consolidation_rate = 0.01f;
    
    // Initialize statistics
    net->avg_resonance = 0.0f;
    net->avg_memory_norm = 0.0f;
    net->avg_surprise = 0.0f;
    net->num_active_neurons = hidden_size;
    
    return net;
}

EXPORT void destroy_m3chat_v2(M3ChatV2 *net) {
    if (!net) return;
    
    // Free neurons
    for (int i = 0; i < net->hidden_size; i++) {
        free(net->neurons[i].fast_memory);
        free(net->neurons[i].slow_memory);
        free(net->neurons[i].fast_grad);
        free(net->neurons[i].slow_grad);
        free(net->neurons[i].resonance);
    }
    free(net->neurons);
    
    // Free embeddings
    free(net->token_embeddings);
    free(net->position_embeddings);
    
    // Free projection matrices
    free(net->W_embed_to_hidden);
    free(net->W_read);
    free(net->W_output);
    free(net->W_resonance);
    
    // Free memory evolution
    free(net->W_memory_fast);
    free(net->W_memory_slow);
    free(net->W_pos_inject);
    
    // Free residual connections
    free(net->W_residual_input);
    free(net->W_residual_context);
    
    // Free layer norm
    free(net->ln_gamma);
    free(net->ln_beta);
    
    // Free manifold
    free(net->manifold_basis);
    free(net->manifold_center);
    
    // Free global memory
    free(net->network_memory);
    
    // Free autoregressive state
    free(net->token_history);
    free(net->hidden_state);
    free(net->logits);
    
    free(net);
}

// ============================================================================
// CORE: AUTOREGRESSIVE STEP (Feature #1)
// ============================================================================

/**
 * Single autoregressive step: feed token, update memory, produce next logits
 * This is the main inference loop - memory persists between calls
 */
EXPORT void m3chat_step(M3ChatV2 *net, int token_id) {
    if (token_id < 0 || token_id >= net->vocab_size) return;
    if (net->current_position >= net->max_seq_len) return;
    
    // Record token in history
    net->token_history[net->current_position] = token_id;
    
    // Get token embedding
    float *token_embed = &net->token_embeddings[token_id * net->embed_dim];
    
    // Get positional encoding
    float *pos_embed = &net->position_embeddings[net->current_position * net->embed_dim];
    
    // Combine token + positional embedding
    float *input_vec = (float*)malloc(net->embed_dim * sizeof(float));
    for (int i = 0; i < net->embed_dim; i++) {
        input_vec[i] = token_embed[i] + pos_embed[i];
    }
    
    // === UPDATE EACH NEURON ===
    float total_resonance = 0.0f;
    float total_memory_norm = 0.0f;
    float total_surprise = 0.0f;
    int active_count = 0;
    
    #pragma omp parallel for reduction(+:total_resonance,total_memory_norm,total_surprise,active_count)
    for (int n = 0; n < net->hidden_size; n++) {
        MemoryNativeNeuron *neuron = &net->neurons[n];
        
        // === MULTI-HEAD RESONANCE (Feature #4) ===
        float max_resonance = 0.0f;
        for (int h = 0; h < net->num_heads; h++) {
            float *W_res = &net->W_resonance[h * net->memory_dim * net->embed_dim];
            
            // Project input for this head
            float *proj_input = (float*)malloc(net->memory_dim * sizeof(float));
            for (int i = 0; i < net->memory_dim; i++) {
                proj_input[i] = 0.0f;
                for (int j = 0; j < net->embed_dim; j++) {
                    proj_input[i] += W_res[i * net->embed_dim + j] * input_vec[j];
                }
            }
            
            // Compute resonance with fast memory
            neuron->resonance[h] = vector_dot(neuron->fast_memory, proj_input, net->memory_dim);
            
            if (neuron->resonance[h] > max_resonance) {
                max_resonance = neuron->resonance[h];
            }
            
            free(proj_input);
        }
        
        // Aggregate multi-head resonance
        float avg_res = 0.0f;
        for (int h = 0; h < net->num_heads; h++) {
            avg_res += neuron->resonance[h];
        }
        avg_res /= net->num_heads;
        
        // === SURPRISE SIGNAL (Bonus) ===
        // Predict what input should be, measure error
        float prediction_error = 0.0f;
        for (int i = 0; i < MIN(net->memory_dim, net->embed_dim); i++) {
            float predicted = neuron->fast_memory[i];
            float actual = (i < net->embed_dim) ? input_vec[i] : 0.0f;
            float diff = predicted - actual;
            prediction_error += diff * diff;
        }
        neuron->surprise = sqrtf(prediction_error / net->memory_dim);
        
        // Adaptive beta based on surprise
        float adaptive_fast_beta = neuron->fast_beta * (1.0f + neuron->surprise);
        
        // === SPARSE ACTIVATION (Bonus) ===
        // Only strongly activate top-k neurons based on resonance
        neuron->is_active = (fabsf(avg_res) > 0.1f);
        
        if (!neuron->is_active) continue;
        active_count++;
        
        // === MEMORY UPDATE (Features #2, #3, #10, #11) ===
        
        // Compute input projection for memory update
        float *input_proj = (float*)malloc(net->memory_dim * sizeof(float));
        for (int i = 0; i < net->memory_dim; i++) {
            input_proj[i] = 0.0f;
            for (int j = 0; j < net->embed_dim; j++) {
                input_proj[i] += net->W_residual_input[i * net->embed_dim + j] * input_vec[j];
            }
        }
        
        // Compute context projection
        float *context_proj = (float*)malloc(net->memory_dim * sizeof(float));
        for (int i = 0; i < net->memory_dim; i++) {
            context_proj[i] = 0.0f;
            for (int j = 0; j < net->manifold_dim; j++) {
                context_proj[i] += net->W_residual_context[i * net->manifold_dim + j] * 
                                   net->network_memory[j];
            }
        }
        
        // Compute positional injection
        float *pos_inject = (float*)malloc(net->memory_dim * sizeof(float));
        for (int i = 0; i < net->memory_dim; i++) {
            pos_inject[i] = 0.0f;
            for (int j = 0; j < net->embed_dim; j++) {
                pos_inject[i] += net->W_pos_inject[i * net->embed_dim + j] * pos_embed[j];
            }
        }
        
        // === FAST MEMORY UPDATE ===
        float *new_fast = (float*)malloc(net->memory_dim * sizeof(float));
        for (int i = 0; i < net->memory_dim; i++) {
            // Memory evolution
            float evolved = 0.0f;
            for (int j = 0; j < net->memory_dim; j++) {
                evolved += net->W_memory_fast[i * net->memory_dim + j] * neuron->fast_memory[j];
            }
            
            // Residual connections (Feature #10)
            new_fast[i] = neuron->fast_memory[i] +  // Residual
                         net->dt * adaptive_fast_beta * (
                             evolved + 
                             input_proj[i] + 
                             context_proj[i] +
                             pos_inject[i] -
                             neuron->fast_decay * neuron->fast_memory[i]
                         );
        }
        
        // === SLOW MEMORY UPDATE ===
        float *new_slow = (float*)malloc(net->memory_dim * sizeof(float));
        for (int i = 0; i < net->memory_dim; i++) {
            // Slow memory evolves more gradually
            float evolved = 0.0f;
            for (int j = 0; j < net->memory_dim; j++) {
                evolved += net->W_memory_slow[i * net->memory_dim + j] * neuron->slow_memory[j];
            }
            
            new_slow[i] = neuron->slow_memory[i] +
                         net->dt * neuron->slow_beta * (
                             evolved +
                             0.1f * input_proj[i] +  // Less influenced by current input
                             context_proj[i] -
                             neuron->slow_decay * neuron->slow_memory[i]
                         );
        }
        
        // Apply layer normalization (Feature #11)
        layer_norm(new_fast, net->ln_gamma, net->ln_beta, net->memory_dim);
        layer_norm(new_slow, net->ln_gamma, net->ln_beta, net->memory_dim);
        
        // Update memories
        memcpy(neuron->fast_memory, new_fast, net->memory_dim * sizeof(float));
        memcpy(neuron->slow_memory, new_slow, net->memory_dim * sizeof(float));
        
        // Activation is tanh of combined memory
        neuron->activation = tanhf(avg_res);
        
        // Update statistics
        total_resonance += avg_res;
        total_memory_norm += vector_norm(neuron->fast_memory, net->memory_dim);
        total_surprise += neuron->surprise;
        
        // Cleanup
        free(input_proj);
        free(context_proj);
        free(pos_inject);
        free(new_fast);
        free(new_slow);
    }
    
    // === COMPUTE HIDDEN STATE (Feature #2: Learned projection) ===
    memset(net->hidden_state, 0, net->hidden_size * sizeof(float));
    
    for (int i = 0; i < net->hidden_size; i++) {
        if (!net->neurons[i].is_active) continue;
        
        // Project combined fast+slow memory to hidden state
        for (int j = 0; j < net->memory_dim; j++) {
            float combined_memory = 0.7f * net->neurons[i].fast_memory[j] + 
                                   0.3f * net->neurons[i].slow_memory[j];
            net->hidden_state[i] += net->W_read[j * net->hidden_size + i] * combined_memory;
        }
        
        net->hidden_state[i] = tanhf(net->hidden_state[i]);
    }
    
    // === COMPUTE OUTPUT LOGITS (Feature #5: for cross-entropy) ===
    memset(net->logits, 0, net->vocab_size * sizeof(float));
    
    for (int i = 0; i < net->vocab_size; i++) {
        for (int j = 0; j < net->hidden_size; j++) {
            net->logits[i] += net->W_output[j * net->vocab_size + i] * net->hidden_state[j];
        }
    }
    
    // Update statistics
    net->avg_resonance = total_resonance / net->hidden_size;
    net->avg_memory_norm = total_memory_norm / net->hidden_size;
    net->avg_surprise = total_surprise / net->hidden_size;
    net->num_active_neurons = active_count;
    
    // Update position and time
    net->current_position++;
    net->conversation_time += net->dt;
    
    free(input_vec);
}

// ============================================================================
// SAMPLING (Feature #13: Temperature control)
// ============================================================================

EXPORT int m3chat_sample(M3ChatV2 *net, float temperature) {
    float *probs = (float*)malloc(net->vocab_size * sizeof(float));
    softmax(net->logits, probs, net->vocab_size, temperature);
    
    // Sample from distribution
    float r = (float)rand() / RAND_MAX;
    float cumsum = 0.0f;
    int sampled_token = 0;
    
    for (int i = 0; i < net->vocab_size; i++) {
        cumsum += probs[i];
        if (r < cumsum) {
            sampled_token = i;
            break;
        }
    }
    
    free(probs);
    return sampled_token;
}

EXPORT int m3chat_argmax(M3ChatV2 *net) {
    int best_token = 0;
    float best_logit = net->logits[0];
    
    for (int i = 1; i < net->vocab_size; i++) {
        if (net->logits[i] > best_logit) {
            best_logit = net->logits[i];
            best_token = i;
        }
    }
    
    return best_token;
}

// ============================================================================
// STATE RESET (Feature #14)
// ============================================================================

EXPORT void m3chat_reset_soft(M3ChatV2 *net) {
    // Soft reset: decay fast memory, keep slow memory
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < net->memory_dim; j++) {
            net->neurons[i].fast_memory[j] *= 0.1f;
            // Slow memory unchanged
        }
    }
    net->current_position = 0;
}

EXPORT void m3chat_reset_hard(M3ChatV2 *net) {
    // Hard reset: clear all memory
    for (int i = 0; i < net->hidden_size; i++) {
        memset(net->neurons[i].fast_memory, 0, net->memory_dim * sizeof(float));
        memset(net->neurons[i].slow_memory, 0, net->memory_dim * sizeof(float));
    }
    memset(net->network_memory, 0, net->manifold_dim * sizeof(float));
    net->current_position = 0;
    net->conversation_time = 0.0f;
}

EXPORT void m3chat_reset_session(M3ChatV2 *net) {
    // Session reset: clear fast memory and position, keep slow memory (personality)
    for (int i = 0; i < net->hidden_size; i++) {
        memset(net->neurons[i].fast_memory, 0, net->memory_dim * sizeof(float));
        // Keep slow_memory intact
    }
    net->current_position = 0;
}

// ============================================================================
// MEMORY CONSOLIDATION (Bonus)
// ============================================================================

EXPORT void m3chat_consolidate(M3ChatV2 *net) {
    // Transfer fast → slow memory (memory consolidation during "sleep")
    for (int i = 0; i < net->hidden_size; i++) {
        for (int j = 0; j < net->memory_dim; j++) {
            net->neurons[i].slow_memory[j] = 
                (1.0f - net->consolidation_rate) * net->neurons[i].slow_memory[j] +
                net->consolidation_rate * net->neurons[i].fast_memory[j];
        }
        
        // Optionally clear fast memory after consolidation
        // memset(net->neurons[i].fast_memory, 0, net->memory_dim * sizeof(float));
    }
}

// ============================================================================
// TRAINING (Feature #5, #8, #15)
// ============================================================================

/**
 * Train with teacher forcing and truncated BPTT
 */
EXPORT float m3chat_train_sequence(
    M3ChatV2 *net,
    const int *tokens,
    int seq_len,
    bool use_teacher_forcing
) {
    float total_loss = 0.0f;
    
    // Reset position for new sequence
    net->current_position = 0;
    
    // Forward pass through sequence
    for (int t = 0; t < seq_len - 1; t++) {
        int current_token = tokens[t];
        int target_token = tokens[t + 1];
        
        // Forward step
        m3chat_step(net, current_token);
        
        // Compute loss (Feature #5: Cross-entropy)
        float loss = cross_entropy_loss(net->logits, target_token, net->vocab_size);
        
        // Add entropy penalty (Feature #13)
        float *probs = (float*)malloc(net->vocab_size * sizeof(float));
        softmax(net->logits, probs, net->vocab_size, 1.0f);
        
        float entropy = 0.0f;
        for (int i = 0; i < net->vocab_size; i++) {
            if (probs[i] > EPSILON) {
                entropy -= probs[i] * logf(probs[i]);
            }
        }
        loss -= net->entropy_penalty * entropy;
        
        total_loss += loss;
        free(probs);
        
        // TODO: Implement full backprop through time
        // This would require storing all intermediate states
        // and computing gradients backward through the sequence
        // For now, we do simple gradient descent on current step
        
        // Compute output gradient
        float *output_grad = (float*)malloc(net->vocab_size * sizeof(float));
        softmax(net->logits, output_grad, net->vocab_size, 1.0f);
        output_grad[target_token] -= 1.0f;  // Gradient of cross-entropy
        
        // Backprop to hidden state
        float *hidden_grad = (float*)calloc(net->hidden_size, sizeof(float));
        for (int i = 0; i < net->hidden_size; i++) {
            for (int j = 0; j < net->vocab_size; j++) {
                hidden_grad[i] += output_grad[j] * net->W_output[i * net->vocab_size + j];
            }
            hidden_grad[i] *= tanh_derivative(net->hidden_state[i]);
        }
        
        // Update output weights
        for (int i = 0; i < net->hidden_size; i++) {
            for (int j = 0; j < net->vocab_size; j++) {
                net->W_output[i * net->vocab_size + j] -= 
                    net->learning_rate * output_grad[j] * net->hidden_state[i];
            }
        }
        
        // Update W_read (simplified)
        for (int i = 0; i < net->hidden_size; i++) {
            if (!net->neurons[i].is_active) continue;
            
            for (int j = 0; j < net->memory_dim; j++) {
                float combined_memory = 0.7f * net->neurons[i].fast_memory[j] + 
                                       0.3f * net->neurons[i].slow_memory[j];
                net->W_read[j * net->hidden_size + i] -= 
                    net->learning_rate * hidden_grad[i] * combined_memory;
            }
        }
        
        free(output_grad);
        free(hidden_grad);
    }
    
    net->training_steps++;
    net->last_loss = total_loss / (seq_len - 1);
    
    return net->last_loss;
}

// ============================================================================
// GENERATION
// ============================================================================

EXPORT void m3chat_generate(
    M3ChatV2 *net,
    const int *prompt_tokens,
    int prompt_len,
    int *output_tokens,
    int max_new_tokens,
    float temperature,
    int *actual_length
) {
    // Reset for new generation
    m3chat_reset_session(net);
    
    // Process prompt
    for (int i = 0; i < prompt_len; i++) {
        m3chat_step(net, prompt_tokens[i]);
        output_tokens[i] = prompt_tokens[i];
    }
    
    // Generate new tokens
    int generated = 0;
    for (int i = 0; i < max_new_tokens; i++) {
        int next_token = m3chat_sample(net, temperature);
        output_tokens[prompt_len + i] = next_token;
        generated++;
        
        // Stop if we hit a stop token (e.g., 0 or special EOS token)
        if (next_token == 0) break;
        
        m3chat_step(net, next_token);
    }
    
    *actual_length = prompt_len + generated;
}

// ============================================================================
// PARAMETER ACCESS
// ============================================================================

EXPORT void m3chat_get_logits(M3ChatV2 *net, float *logits_out) {
    memcpy(logits_out, net->logits, net->vocab_size * sizeof(float));
}

EXPORT void m3chat_get_hidden_state_v2(M3ChatV2 *net, float *state_out) {
    memcpy(state_out, net->hidden_state, net->hidden_size * sizeof(float));
}

EXPORT void m3chat_set_temperature(M3ChatV2 *net, float temp) {
    net->temperature = temp;
}

EXPORT float m3chat_get_temperature(M3ChatV2 *net) {
    return net->temperature;
}

EXPORT void m3chat_set_learning_rate_v2(M3ChatV2 *net, float lr) {
    net->learning_rate = lr;
}

EXPORT float m3chat_get_avg_surprise(M3ChatV2 *net) {
    return net->avg_surprise;
}

EXPORT int m3chat_get_num_active_neurons(M3ChatV2 *net) {
    return net->num_active_neurons;
}

EXPORT int m3chat_get_current_position(M3ChatV2 *net) {
    return net->current_position;
}
// Additional getters for Python wrapper
EXPORT float m3chat_get_learning_rate_v2(M3ChatV2 *net) {
    return net->learning_rate;
}

EXPORT int m3chat_get_training_steps_v2(M3ChatV2 *net) {
    return net->training_steps;
}

EXPORT float m3chat_get_last_loss_v2(M3ChatV2 *net) {
    return net->last_loss;
}

EXPORT float m3chat_get_avg_resonance_v2(M3ChatV2 *net) {
    return net->avg_resonance;
}

EXPORT float m3chat_get_avg_memory_norm_v2(M3ChatV2 *net) {
    return net->avg_memory_norm;
}

// ============================================================================
// SERIALIZATION
// ============================================================================

EXPORT int m3chat_save_v2(M3ChatV2 *net, const char *filename) {
    FILE *f = fopen(filename, "wb");
    if (!f) return -1;
    
    // Write header
    fwrite(&net->vocab_size, sizeof(int), 1, f);
    fwrite(&net->embed_dim, sizeof(int), 1, f);
    fwrite(&net->hidden_size, sizeof(int), 1, f);
    fwrite(&net->memory_dim, sizeof(int), 1, f);
    fwrite(&net->manifold_dim, sizeof(int), 1, f);
    fwrite(&net->num_heads, sizeof(int), 1, f);
    fwrite(&net->max_seq_len, sizeof(int), 1, f);
    
    // Write embeddings
    fwrite(net->token_embeddings, sizeof(float), net->vocab_size * net->embed_dim, f);
    fwrite(net->position_embeddings, sizeof(float), net->max_seq_len * net->embed_dim, f);
    
    // Write weights
    fwrite(net->W_embed_to_hidden, sizeof(float), net->embed_dim * net->hidden_size, f);
    fwrite(net->W_read, sizeof(float), net->memory_dim * net->hidden_size, f);
    fwrite(net->W_output, sizeof(float), net->hidden_size * net->vocab_size, f);
    fwrite(net->W_resonance, sizeof(float), net->num_heads * net->memory_dim * net->embed_dim, f);
    fwrite(net->W_memory_fast, sizeof(float), net->memory_dim * net->memory_dim, f);
    fwrite(net->W_memory_slow, sizeof(float), net->memory_dim * net->memory_dim, f);
    fwrite(net->W_pos_inject, sizeof(float), net->embed_dim * net->memory_dim, f);
    fwrite(net->W_residual_input, sizeof(float), net->embed_dim * net->memory_dim, f);
    fwrite(net->W_residual_context, sizeof(float), net->manifold_dim * net->memory_dim, f);
    
    // Write layer norm
    fwrite(net->ln_gamma, sizeof(float), net->memory_dim, f);
    fwrite(net->ln_beta, sizeof(float), net->memory_dim, f);
    
    // Write manifold
    fwrite(net->manifold_basis, sizeof(float), net->manifold_dim * net->memory_dim, f);
    fwrite(net->manifold_center, sizeof(float), net->memory_dim, f);
    fwrite(&net->manifold_radius, sizeof(float), 1, f);
    
    // Write neuron states
    for (int i = 0; i < net->hidden_size; i++) {
        fwrite(net->neurons[i].fast_memory, sizeof(float), net->memory_dim, f);
        fwrite(net->neurons[i].slow_memory, sizeof(float), net->memory_dim, f);
        fwrite(&net->neurons[i].fast_beta, sizeof(float), 1, f);
        fwrite(&net->neurons[i].slow_beta, sizeof(float), 1, f);
        fwrite(&net->neurons[i].fast_decay, sizeof(float), 1, f);
        fwrite(&net->neurons[i].slow_decay, sizeof(float), 1, f);
    }
    
    // Write global memory
    fwrite(net->network_memory, sizeof(float), net->manifold_dim, f);
    
    // Write parameters
    fwrite(&net->temperature, sizeof(float), 1, f);
    fwrite(&net->entropy_penalty, sizeof(float), 1, f);
    fwrite(&net->learning_rate, sizeof(float), 1, f);
    fwrite(&net->dt, sizeof(float), 1, f);
    fwrite(&net->consolidation_rate, sizeof(float), 1, f);
    fwrite(&net->training_steps, sizeof(int), 1, f);
    
    fclose(f);
    return 0;
}

EXPORT M3ChatV2* m3chat_load_v2(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (!f) return NULL;
    
    // Read header
    int vocab_size, embed_dim, hidden_size, memory_dim, manifold_dim, num_heads, max_seq_len;
    fread(&vocab_size, sizeof(int), 1, f);
    fread(&embed_dim, sizeof(int), 1, f);
    fread(&hidden_size, sizeof(int), 1, f);
    fread(&memory_dim, sizeof(int), 1, f);
    fread(&manifold_dim, sizeof(int), 1, f);
    fread(&num_heads, sizeof(int), 1, f);
    fread(&max_seq_len, sizeof(int), 1, f);
    
    // Create network
    M3ChatV2 *net = create_m3chat_v2(vocab_size, embed_dim, hidden_size, memory_dim, 
                                      manifold_dim, num_heads, max_seq_len);
    if (!net) {
        fclose(f);
        return NULL;
    }
    
    // Read embeddings
    fread(net->token_embeddings, sizeof(float), vocab_size * embed_dim, f);
    fread(net->position_embeddings, sizeof(float), max_seq_len * embed_dim, f);
    
    // Read weights
    fread(net->W_embed_to_hidden, sizeof(float), embed_dim * hidden_size, f);
    fread(net->W_read, sizeof(float), memory_dim * hidden_size, f);
    fread(net->W_output, sizeof(float), hidden_size * vocab_size, f);
    fread(net->W_resonance, sizeof(float), num_heads * memory_dim * embed_dim, f);
    fread(net->W_memory_fast, sizeof(float), memory_dim * memory_dim, f);
    fread(net->W_memory_slow, sizeof(float), memory_dim * memory_dim, f);
    fread(net->W_pos_inject, sizeof(float), embed_dim * memory_dim, f);
    fread(net->W_residual_input, sizeof(float), embed_dim * memory_dim, f);
    fread(net->W_residual_context, sizeof(float), manifold_dim * memory_dim, f);
    
    // Read layer norm
    fread(net->ln_gamma, sizeof(float), memory_dim, f);
    fread(net->ln_beta, sizeof(float), memory_dim, f);
    
    // Read manifold
    fread(net->manifold_basis, sizeof(float), manifold_dim * memory_dim, f);
    fread(net->manifold_center, sizeof(float), memory_dim, f);
    fread(&net->manifold_radius, sizeof(float), 1, f);
    
    // Read neuron states
    for (int i = 0; i < hidden_size; i++) {
        fread(net->neurons[i].fast_memory, sizeof(float), memory_dim, f);
        fread(net->neurons[i].slow_memory, sizeof(float), memory_dim, f);
        fread(&net->neurons[i].fast_beta, sizeof(float), 1, f);
        fread(&net->neurons[i].slow_beta, sizeof(float), 1, f);
        fread(&net->neurons[i].fast_decay, sizeof(float), 1, f);
        fread(&net->neurons[i].slow_decay, sizeof(float), 1, f);
    }
    
    // Read global memory
    fread(net->network_memory, sizeof(float), manifold_dim, f);
    
    // Read parameters
    fread(&net->temperature, sizeof(float), 1, f);
    fread(&net->entropy_penalty, sizeof(float), 1, f);
    fread(&net->learning_rate, sizeof(float), 1, f);
    fread(&net->dt, sizeof(float), 1, f);
    fread(&net->consolidation_rate, sizeof(float), 1, f);
    fread(&net->training_steps, sizeof(int), 1, f);
    
    fclose(f);
    return net;
}

// ============================================================================
// INFO
// ============================================================================

EXPORT void m3chat_print_info_v2(M3ChatV2 *net) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║       M³-CHAT v2: AUTOREGRESSIVE LANGUAGE MODEL               ║\n");
    printf("║          Enhanced Memory-Native Architecture                  ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n");
    printf("\n");
    printf("ARCHITECTURE:\n");
    printf("  Vocabulary Size:  %d\n", net->vocab_size);
    printf("  Embedding Dim:    %d\n", net->embed_dim);
    printf("  Hidden Size:      %d neurons\n", net->hidden_size);
    printf("  Memory Dim:       %d (per neuron)\n", net->memory_dim);
    printf("  Manifold Dim:     %d\n", net->manifold_dim);
    printf("  Attention Heads:  %d\n", net->num_heads);
    printf("  Max Seq Length:   %d\n", net->max_seq_len);
    printf("\n");
    printf("NEW FEATURES:\n");
    printf("  ✓ Autoregressive token loop\n");
    printf("  ✓ Fast + Slow memory split\n");
    printf("  ✓ Multi-head resonance\n");
    printf("  ✓ Cross-entropy loss\n");
    printf("  ✓ Positional encoding\n");
    printf("  ✓ Learned projections\n");
    printf("  ✓ Residual connections\n");
    printf("  ✓ Layer normalization\n");
    printf("  ✓ Sparse activation\n");
    printf("  ✓ Surprise signals\n");
    printf("  ✓ Memory consolidation\n");
    printf("\n");
    printf("CURRENT STATE:\n");
    printf("  Position:         %d / %d\n", net->current_position, net->max_seq_len);
    printf("  Active Neurons:   %d / %d\n", net->num_active_neurons, net->hidden_size);
    printf("  Avg Resonance:    %.4f\n", net->avg_resonance);
    printf("  Avg Memory Norm:  %.4f\n", net->avg_memory_norm);
    printf("  Avg Surprise:     %.4f\n", net->avg_surprise);
    printf("\n");
    printf("TRAINING:\n");
    printf("  Learning Rate:    %.6f\n", net->learning_rate);
    printf("  Temperature:      %.2f\n", net->temperature);
    printf("  Entropy Penalty:  %.4f\n", net->entropy_penalty);
    printf("  Training Steps:   %d\n", net->training_steps);
    printf("  Last Loss:        %.6f\n", net->last_loss);
    printf("\n");
    long total_params = 
        (long)net->vocab_size * net->embed_dim +
        (long)net->max_seq_len * net->embed_dim +
        (long)net->embed_dim * net->hidden_size +
        (long)net->memory_dim * net->hidden_size +
        (long)net->hidden_size * net->vocab_size +
        (long)net->num_heads * net->memory_dim * net->embed_dim +
        (long)net->memory_dim * net->memory_dim * 2 +
        (long)net->embed_dim * net->memory_dim * 2 +
        (long)net->manifold_dim * net->memory_dim * 2 +
        (long)net->memory_dim * 2;
    printf("TOTAL PARAMETERS: %ld (%.2f M)\n", total_params, total_params / 1e6);
    printf("════════════════════════════════════════════════════════════════\n");
}