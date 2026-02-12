import math
import mlx.core as mx
import mlx.nn as nn

# --- Phase 1: Configuration ---
# On utilise des noms explicites pour que tout le monde comprenne
class GPTConfig:
    context_window: int = 128  # Taille max de la séquence (ex: 128 mots)
    vocab_size: int = 65       # Nombre de caractères uniques possibles
    num_layers: int = 4        # Nombre de blocs Transformer empilés
    num_heads: int = 4         # Nombre de "cerveaux" d'attention parallèles
    embedding_dim: int = 128   # Taille du vecteur pour chaque mot
    dropout: float = 0.0
    bias: bool = True          # Active les biais dans les calculs

config = GPTConfig()


# --- Phase 2: Le Cerveau (Self-Attention) ---
class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.embedding_dim % config.num_heads == 0
        
        self.num_heads = config.num_heads
        self.head_dim = config.embedding_dim // config.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Une seule grosse couche linéaire pour calculer Query, Key, Value d'un coup
        # Taille: 128 -> 3 * 128 (384)
        self.projection_qkv = nn.Linear(config.embedding_dim, 3 * config.embedding_dim, bias=config.bias)
        
        # Couche de sortie pour réassembler les résultats
        self.output_projection = nn.Linear(config.embedding_dim, config.embedding_dim, bias=config.bias)

    def __call__(self, x, mask=None):
        B, T, C = x.shape # Batch, Time (Sequence), Channels (Embeddings)
        
        # 1. On calcule Q, K, V
        qkv = self.projection_qkv(x)
        queries, keys, values = mx.split(qkv, 3, axis=-1)
        
        # 2. On découpe pour le Multi-Head Attention
        # On passe de (B, T, 128) à (B, 4, T, 32) pour avoir 4 têtes indépendantes
        queries = queries.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, T, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 3. Calcul du score d'attention (Qui regarde qui ?)
        # Formule : softmax( (Q @ K.T) / sqrt(dim) )
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        
        # 4. Masque Causal (Interdiction de regarder le futur)
        # On remplit le triangle supérieur avec -infini
        indices = mx.arange(T)
        mask = indices[:, None] < indices[None, :]
        scores = mx.where(mask, -1e9, scores)
        
        # 5. Probabilités finales
        probs = mx.softmax(scores, axis=-1)
        
        # 6. Agrégation de l'information (Attn @ V)
        context = probs @ values 
        
        # 7. Réassemblage des têtes (Concaténation)
        context = context.transpose(0, 2, 1, 3).reshape(B, T, C)
        
        return self.output_projection(context)


# --- Phase 3: Réflexion (MLP / Feed Forward) ---
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        # On étend la dimension x4 pour "réfléchir"
        self.expand = nn.Linear(config.embedding_dim, 4 * config.embedding_dim, bias=config.bias)
        self.activation = nn.GELU() 
        # On revient à la dimension normale
        self.contract = nn.Linear(4 * config.embedding_dim, config.embedding_dim, bias=config.bias)

    def __call__(self, x):
        x = self.expand(x)
        x = self.activation(x)
        x = self.contract(x)
        return x


# --- Phase 4: Le Bloc Transformer ---
class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embedding_dim)
        self.attention = CausalSelfAttention(config)
        
        self.norm2 = nn.LayerNorm(config.embedding_dim)
        self.mlp = MLP(config)

    def __call__(self, x):
        # Connexions Résiduelles (Skip Connections) : x + ...
        # Crucial pour éviter que le gradient ne disparaisse
        x = x + self.attention(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# --- Phase 5: Le Modèle GPT Complet ---
class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Tables d'Embeddings
        self.token_embeddings = nn.Embedding(config.vocab_size, config.embedding_dim)
        self.position_embeddings = nn.Embedding(config.context_window, config.embedding_dim)
        
        # Empilement des blocs
        self.blocks = [Block(config) for _ in range(config.num_layers)]
        
        # Couche finale
        self.final_norm = nn.LayerNorm(config.embedding_dim)
        self.language_head = nn.Linear(config.embedding_dim, config.vocab_size, bias=False)
        
        # Partage de poids (Optimisation standard)
        self.language_head.weight = self.token_embeddings.weight 

    def __call__(self, indices):
        B, T = indices.shape
        
        # On crée les vecteurs de position [0, 1, 2, ..., T-1]
        positions = mx.arange(0, T, dtype=mx.int32)
        
        # Combinaison Token + Position
        x = self.token_embeddings(indices) + self.position_embeddings(positions)
        
        # Passage dans les blocs Transformer
        for block in self.blocks:
            x = block(x)
            
        x = self.final_norm(x)
        logits = self.language_head(x)
        
        return logits

    def generate(self, indices, max_new_tokens):
        # Boucle de génération simple (pour tester)
        for _ in range(max_new_tokens):
            # On garde juste les derniers tokens pour ne pas dépasser la mémoire
            indices_cond = indices[:, -self.config.context_window:]
            
            # Prédiction
            logits = self(indices_cond)
            logits = logits[:, -1, :] # On prend juste le dernier token
            
            # Choix du token le plus probable (Greedy)
            next_token = mx.argmax(logits, axis=-1, keepdims=True)
            
            # Ajout à la suite
            indices = mx.concatenate([indices, next_token], axis=1)
            
        return indices


# --- Phase 6: Test Rapide ---
# 1. Initialisation
model = GPT(config)
mx.eval(model.parameters()) 

print("Modèle initialisé sur :", mx.default_device())

# 2. Création d'une fausse phrase (indices aléatoires)
import numpy as np
fake_input = mx.array(np.random.randint(0, config.vocab_size, (1, 10)))

# 3. Test de prédiction
output = model(fake_input)
print(f"Forme de la sortie (Batch, Time, Vocab) : {output.shape}") 

# 4. Test de génération
generated = model.generate(fake_input, max_new_tokens=20)
print(f"Longueur séquence générée : {generated.shape[1]}")